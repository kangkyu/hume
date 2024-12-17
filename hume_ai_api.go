package hume

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

// Client handles communication with Hume AI API
type Client struct {
	apiKey     string
	baseURL    string
	wsURL      string
	mu         sync.Mutex
	wsConn     *websocket.Conn
	httpClient *http.Client
}

// ClientOption allows customizing the client
type ClientOption func(*Client)

// NewClient creates a new Hume AI client
func NewClient(apiKey string, opts ...ClientOption) *Client {
	c := &Client{
		apiKey:  apiKey,
		baseURL: "https://api.hume.ai/v0",
		wsURL:   "wss://api.hume.ai/v0",
		httpClient: &http.Client{
			Timeout: time.Second * 30,
		},
	}

	for _, opt := range opts {
		opt(c)
	}

	return c
}

type VoiceChatConfig struct {
	SampleRate    int
	NumChannels   int
	BitsPerSample int
	ModelName     string
	LanguageCode  string
}

// VoiceChatResponse represents a response from the voice chat
type VoiceChatResponse struct {
	Type     string             `json:"type"`
	Text     string             `json:"text,omitempty"`
	Emotions map[string]float64 `json:"emotions,omitempty"`
	Error    string             `json:"error,omitempty"`
	IsFinal  bool               `json:"is_final,omitempty"`
}

// VoiceChatHandler handles voice chat events
type VoiceChatHandler interface {
	OnConnect()
	OnDisconnect(error)
	OnResponse(VoiceChatResponse)
}

type defaultHandler struct{}

func (h *defaultHandler) OnConnect()                   {}
func (h *defaultHandler) OnDisconnect(error)           {}
func (h *defaultHandler) OnResponse(VoiceChatResponse) {}

// ChatMessage represents a message in the chat
type ChatMessage struct {
	Type    string `json:"type"`
	Content string `json:"content,omitempty"`
	Error   string `json:"error,omitempty"`
}

// ChatHandler handles chat events
type ChatHandler interface {
	OnConnect()
	OnMessage(ChatMessage)
	OnError(error)
	OnClose()
}

// defaultChatHandler provides default implementations
type defaultChatHandler struct{}

func (h *defaultChatHandler) OnConnect()            {}
func (h *defaultChatHandler) OnMessage(ChatMessage) {}
func (h *defaultChatHandler) OnError(error)         {}
func (h *defaultChatHandler) OnClose()              {}

// StartChat initiates a chat session with a specific config
func (c *Client) StartChat(ctx context.Context, configID string, handler ChatHandler) error {
	if handler == nil {
		handler = &defaultChatHandler{}
	}

	// Construct WebSocket URL
	u, err := url.Parse(strings.Replace(c.baseURL, "https://", "wss://", 1) + "/evi/chat")
	if err != nil {
		return fmt.Errorf("parsing WebSocket URL: %w", err)
	}

	// Add query parameters
	q := u.Query()
	q.Set("config_id", configID)
	u.RawQuery = q.Encode()

	// Set up WebSocket connection
	headers := http.Header{}
	headers.Set("X-Hume-Api-Key", c.apiKey)

	dialer := websocket.Dialer{
		HandshakeTimeout: 10 * time.Second,
	}

	conn, _, err := dialer.DialContext(ctx, u.String(), headers)
	if err != nil {
		return fmt.Errorf("connecting to WebSocket: %w", err)
	}

	c.mu.Lock()
	c.wsConn = conn
	c.mu.Unlock()

	handler.OnConnect()

	// Start reading messages
	go c.readChatMessages(ctx, handler)

	return nil
}

// SendChatMessage sends a message to the chat
func (c *Client) SendChatMessage(message string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.wsConn == nil {
		return fmt.Errorf("no active chat session")
	}

	msg := ChatMessage{
		Type:    "message",
		Content: message,
	}

	return c.wsConn.WriteJSON(msg)
}

// StopChat ends the chat session
func (c *Client) StopChat() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.wsConn == nil {
		return nil
	}

	err := c.wsConn.WriteMessage(
		websocket.CloseMessage,
		websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""),
	)
	if err != nil {
		return fmt.Errorf("sending close message: %w", err)
	}

	err = c.wsConn.Close()
	c.wsConn = nil
	return err
}

func (c *Client) readChatMessages(ctx context.Context, handler ChatHandler) {
	defer func() {
		c.mu.Lock()
		if c.wsConn != nil {
			c.wsConn.Close()
			c.wsConn = nil
		}
		c.mu.Unlock()
		handler.OnClose()
	}()

	for {
		select {
		case <-ctx.Done():
			handler.OnError(ctx.Err())
			return
		default:
			var message ChatMessage
			err := c.wsConn.ReadJSON(&message)
			if err != nil {
				if websocket.IsCloseError(err, websocket.CloseNormalClosure) {
					return
				}
				handler.OnError(err)
				return
			}

			handler.OnMessage(message)
		}
	}
}

// Prompt represents the configuration prompt
type Prompt struct {
	ID                 string `json:"id"`
	Version            int    `json:"version"`
	VersionType        string `json:"version_type"`
	Name               string `json:"name"`
	CreatedOn          int64  `json:"created_on"`
	ModifiedOn         int64  `json:"modified_on"`
	Text               string `json:"text"`
	VersionDescription string `json:"version_description"`
}

// VoiceParameters represents the custom voice parameters
type VoiceParameters struct {
	Gender        int `json:"gender"`
	Assertiveness int `json:"assertiveness"`
	Buoyancy      int `json:"buoyancy"`
	Confidence    int `json:"confidence"`
	Enthusiasm    int `json:"enthusiasm"`
	Nasality      int `json:"nasality"`
	Relaxedness   int `json:"relaxedness"`
	Smoothness    int `json:"smoothness"`
	Tepidity      int `json:"tepidity"`
	Tightness     int `json:"tightness"`
}

// CustomVoice represents the custom voice configuration
type CustomVoice struct {
	ID             string          `json:"id"`
	Version        int             `json:"version"`
	Name           string          `json:"name"`
	CreatedOn      int64           `json:"created_on"`
	ModifiedOn     int64           `json:"modified_on"`
	BaseVoice      string          `json:"base_voice"`
	ParameterModel string          `json:"parameter_model"`
	Parameters     VoiceParameters `json:"parameters"`
}

// Voice represents the voice configuration
type Voice struct {
	Provider    string      `json:"provider"`
	Name        string      `json:"name"`
	CustomVoice CustomVoice `json:"custom_voice"`
}

// LanguageModel represents the language model configuration
type LanguageModel struct {
	ModelProvider string  `json:"model_provider"`
	ModelResource string  `json:"model_resource"`
	Temperature   float64 `json:"temperature"`
}

// ELLMModel represents the ELLM model configuration
type ELLMModel struct {
	AllowShortResponses bool `json:"allow_short_responses"`
}

// EventMessage represents an event message configuration
type EventMessage struct {
	Enabled bool   `json:"enabled"`
	Text    string `json:"text"`
}

// EventMessages represents all event message configurations
type EventMessages struct {
	OnNewChat            EventMessage `json:"on_new_chat"`
	OnInactivityTimeout  EventMessage `json:"on_inactivity_timeout"`
	OnMaxDurationTimeout EventMessage `json:"on_max_duration_timeout"`
}

// Timeout represents a timeout configuration
type Timeout struct {
	Enabled      bool `json:"enabled"`
	DurationSecs int  `json:"duration_secs"`
}

// Timeouts represents all timeout configurations
type Timeouts struct {
	Inactivity  Timeout `json:"inactivity"`
	MaxDuration Timeout `json:"max_duration"`
}

// Config represents an AI configuration
type Config struct {
	ID                 string        `json:"id"`
	Version            int           `json:"version"`
	EviVersion         string        `json:"evi_version"`
	VersionDescription string        `json:"version_description"`
	Name               string        `json:"name"`
	CreatedOn          int64         `json:"created_on"`
	ModifiedOn         int64         `json:"modified_on"`
	Prompt             Prompt        `json:"prompt"`
	Voice              Voice         `json:"voice"`
	LanguageModel      LanguageModel `json:"language_model"`
	ELLMModel          ELLMModel     `json:"ellm_model"`
	Tools              []interface{} `json:"tools"`
	BuiltinTools       []interface{} `json:"builtin_tools"`
	EventMessages      EventMessages `json:"event_messages"`
	Timeouts           Timeouts      `json:"timeouts"`
}

// ListConfigsResponse represents the response from listing configs
type ListConfigsResponse struct {
	TotalPages  int      `json:"total_pages"`
	PageNumber  int      `json:"page_number"`
	PageSize    int      `json:"page_size"`
	ConfigsPage []Config `json:"configs_page"`
}

// ListConfigsParams represents optional parameters for listing configs
type ListConfigsParams struct {
	PageNumber int `json:"page_number,omitempty"`
	PageSize   int `json:"page_size,omitempty"`
}

// ListConfigs retrieves a list of AI configurations
func (c *Client) ListConfigs(ctx context.Context, params *ListConfigsParams) (*ListConfigsResponse, error) {
	endpoint := fmt.Sprintf("%s/evi/configs", c.baseURL)

	req, err := http.NewRequestWithContext(ctx, "GET", endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}

	// Add query parameters if provided
	if params != nil {
		q := req.URL.Query()
		if params.PageNumber >= 0 {
			q.Set("page_number", fmt.Sprintf("%d", params.PageNumber))
		}
		if params.PageSize > 0 {
			q.Set("page_size", fmt.Sprintf("%d", params.PageSize))
		}
		req.URL.RawQuery = q.Encode()
	}

	req.Header.Set("X-Hume-Api-Key", c.apiKey)
	req.Header.Set("Accept", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("making request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error: status=%d body=%s", resp.StatusCode, string(body))
	}

	var result ListConfigsResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decoding response: %w", err)
	}

	return &result, nil
}

// ChatConfig represents the configuration of a chat
type ChatConfig struct {
	ID      string `json:"id"`
	Version int    `json:"version"`
}

// Chat represents a chat session
type Chat struct {
	ID             string     `json:"id"`
	ChatGroupID    string     `json:"chat_group_id"`
	Status         string     `json:"status"`
	StartTimestamp int64      `json:"start_timestamp"`
	EndTimestamp   int64      `json:"end_timestamp"`
	EventCount     int        `json:"event_count"`
	Metadata       string     `json:"metadata"`
	Config         ChatConfig `json:"config"`
}

// ListChatsResponse represents the response from listing chats
type ListChatsResponse struct {
	PageNumber          int    `json:"page_number"`
	PageSize            int    `json:"page_size"`
	TotalPages          int    `json:"total_pages"`
	PaginationDirection string `json:"pagination_direction"`
	ChatsPage           []Chat `json:"chats_page"`
}

// ListChatsParams represents optional parameters for listing chats
type ListChatsParams struct {
	PageNumber     int  `json:"page_number,omitempty"`
	PageSize       int  `json:"page_size,omitempty"`
	AscendingOrder bool `json:"ascending_order,omitempty"`
}

// ListChats retrieves a list of chat sessions
func (c *Client) ListChats(ctx context.Context, params *ListChatsParams) (*ListChatsResponse, error) {
	endpoint := fmt.Sprintf("%s/evi/chats", c.baseURL)

	req, err := http.NewRequestWithContext(ctx, "GET", endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}

	// Add query parameters if provided
	if params != nil {
		q := req.URL.Query()
		if params.PageNumber >= 0 {
			q.Set("page_number", fmt.Sprintf("%d", params.PageNumber))
		}
		if params.PageSize > 0 {
			q.Set("page_size", fmt.Sprintf("%d", params.PageSize))
		}
		q.Set("ascending_order", fmt.Sprintf("%t", params.AscendingOrder))
		req.URL.RawQuery = q.Encode()
	}

	req.Header.Set("X-Hume-Api-Key", c.apiKey)
	req.Header.Set("Accept", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("making request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error: status=%d body=%s", resp.StatusCode, string(body))
	}

	var result ListChatsResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decoding response: %w", err)
	}

	return &result, nil
}

// StartVoiceChat initiates a voice chat session
func (c *Client) StartVoiceChat(ctx context.Context, config VoiceChatConfig, handler VoiceChatHandler) error {
	if handler == nil {
		handler = &defaultHandler{}
	}

	c.mu.Lock()
	if c.wsConn != nil {
		c.mu.Unlock()
		return fmt.Errorf("voice chat session already active")
	}

	// Construct WebSocket URL with parameters
	u, err := url.Parse(c.wsURL + "/evi/chat")
	if err != nil {
		c.mu.Unlock()
		return fmt.Errorf("parsing WebSocket URL: %w", err)
	}

	q := u.Query()
	q.Set("sample_rate", fmt.Sprintf("%d", config.SampleRate))
	q.Set("channels", fmt.Sprintf("%d", config.NumChannels))
	q.Set("bits_per_sample", fmt.Sprintf("%d", config.BitsPerSample))
	if config.ModelName != "" {
		q.Set("model", config.ModelName)
	}
	if config.LanguageCode != "" {
		q.Set("language", config.LanguageCode)
	}
	u.RawQuery = q.Encode()

	log.Printf("Attempting WebSocket connection to: %s", u.String())

	headers := http.Header{}
	headers.Set("Authorization", "Bearer "+c.apiKey)
	headers.Set("X-Hume-Api-Key", c.apiKey)

	// Connect to WebSocket
	dialer := websocket.Dialer{
		HandshakeTimeout: 15 * time.Second, // Increased timeout
		ReadBufferSize:   1024,
		WriteBufferSize:  1024,
	}

	conn, resp, err := dialer.DialContext(ctx, u.String(), headers)
	if err != nil {
		// Log full response details for debugging
		if resp != nil {
			body, _ := io.ReadAll(resp.Body)
			log.Printf("Response Status: %s", resp.Status)
			log.Printf("Response Body: %s", string(body))
		}
		return fmt.Errorf("websocket connection failed: %w", err)
	}

	c.wsConn = conn
	c.mu.Unlock()

	handler.OnConnect()

	// Start reading responses
	go c.readResponses(ctx, handler)

	return nil
}

// SendAudioData sends audio data over the WebSocket connection
func (c *Client) SendAudioData(data []byte) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.wsConn == nil {
		return fmt.Errorf("no active voice chat session")
	}

	return c.wsConn.WriteMessage(websocket.BinaryMessage, data)
}

// StopVoiceChat ends the voice chat session
func (c *Client) StopVoiceChat() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.wsConn == nil {
		return nil
	}

	err := c.wsConn.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""))
	if err != nil {
		return fmt.Errorf("sending close message: %w", err)
	}

	err = c.wsConn.Close()
	c.wsConn = nil
	return err
}

func (c *Client) readResponses(ctx context.Context, handler VoiceChatHandler) {
	defer func() {
		c.mu.Lock()
		if c.wsConn != nil {
			c.wsConn.Close()
			c.wsConn = nil
		}
		c.mu.Unlock()
	}()

	for {
		select {
		case <-ctx.Done():
			handler.OnDisconnect(ctx.Err())
			return
		default:
			_, message, err := c.wsConn.ReadMessage()
			if err != nil {
				handler.OnDisconnect(err)
				return
			}

			var response VoiceChatResponse
			if err := json.Unmarshal(message, &response); err != nil {
				handler.OnDisconnect(fmt.Errorf("parsing response: %w", err))
				return
			}

			handler.OnResponse(response)
		}
	}
}

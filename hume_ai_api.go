package hume

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
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
		baseURL: "https://api.hume.ai/v1",
		wsURL:   "wss://api.hume.ai/v1/ws/voice",
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

// Chat represents a chat session
type Chat struct {
	ID        string    `json:"id"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
	Status    string    `json:"status"`
}

// ListChatsResponse represents the response from listing chats
type ListChatsResponse struct {
	Chats []Chat `json:"chats"`
	Total int    `json:"total"`
}

// ListChatsParams represents optional parameters for listing chats
type ListChatsParams struct {
	Page     int    `json:"page,omitempty"`
	PageSize int    `json:"page_size,omitempty"`
	Status   string `json:"status,omitempty"`
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
		if params.Page > 0 {
			q.Set("page", fmt.Sprintf("%d", params.Page))
		}
		if params.PageSize > 0 {
			q.Set("page_size", fmt.Sprintf("%d", params.PageSize))
		}
		if params.Status != "" {
			q.Set("status", params.Status)
		}
		req.URL.RawQuery = q.Encode()
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.apiKey))
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
	u, err := url.Parse(c.wsURL)
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

	// Connect to WebSocket
	dialer := websocket.Dialer{
		HandshakeTimeout: 10 * time.Second,
	}

	headers := make(map[string][]string)
	headers["Authorization"] = []string{"Bearer " + c.apiKey}

	conn, _, err := dialer.Dial(u.String(), headers)
	if err != nil {
		c.mu.Unlock()
		return fmt.Errorf("connecting to WebSocket: %w", err)
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

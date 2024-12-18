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

type VoiceChatResponse interface {
	GetType() string
}

type ChatMetadata struct {
	Type        string `json:"type"`
	ChatGroupID string `json:"chat_group_id"`
	ChatID      string `json:"chat_id"`
}

func (c ChatMetadata) GetType() string { return c.Type }

type AssistantMessage struct {
	Type    string  `json:"type"`
	Message Message `json:"message"`
	//Models   Models  `json:"models"`
	FromText bool `json:"from_text"`
}

func (a AssistantMessage) GetType() string { return a.Type }

type AudioResponse struct {
	Type            string `json:"type"`
	ID              string `json:"id"`
	Index           int    `json:"index"`
	Data            string `json:"data"`
	CustomSessionId string `json:"custom_session_id,omitempty"`

	RawMessage json.RawMessage `json:"-"` // For debugging
}

func (a AudioResponse) GetType() string { return a.Type }

type AssistantEnd struct {
	Type string `json:"type"`
}

func (a AssistantEnd) GetType() string { return a.Type }

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	// Add fields according to the Message object structure
}

// Client handles communication with Hume AI API
type Client struct {
	apiKey     string
	baseURL    string
	mu         sync.RWMutex
	wsConn     *websocket.Conn
	httpClient *http.Client
	isActive   bool
}

// ClientOption allows customizing the client
type ClientOption func(*Client)

// NewClient creates a new Hume AI client
func NewClient(apiKey string, opts ...ClientOption) *Client {
	c := &Client{
		apiKey:  apiKey,
		baseURL: "https://api.hume.ai/v0",
		httpClient: &http.Client{
			Timeout: time.Second * 30,
		},
	}

	for _, opt := range opts {
		opt(c)
	}

	return c
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

// StartVoiceChat initiates a voice chat session
func (c *Client) StartVoiceChat(ctx context.Context, configID string, handler VoiceChatHandler) error {
	if handler == nil {
		handler = &defaultHandler{}
	}
	// Add logging
	log.Printf("Starting voice chat with config ID: %s", configID)

	c.mu.Lock()
	if c.wsConn != nil {
		c.mu.Unlock()
		return fmt.Errorf("voice chat session already active")
	}

	// Build WebSocket URL
	u, err := url.Parse(strings.Replace(c.baseURL, "https://", "wss://", 1) + "/evi/chat")
	if err != nil {
		c.mu.Unlock()
		return fmt.Errorf("parsing WebSocket URL: %w", err)
	}

	// Prepare query parameters
	q := u.Query()
	q.Set("config_id", configID)

	// Add chat_group_id if available in context
	if chatGroupID, ok := ctx.Value("chat_group_id").(string); ok && chatGroupID != "" {
		q.Set("resumed_chat_group_id", chatGroupID)
		log.Printf("Resuming chat with group ID: %s", chatGroupID)
	}

	u.RawQuery = q.Encode()

	log.Printf("Attempting WebSocket connection to: %s", u.String())

	headers := http.Header{}
	headers.Set("X-Hume-Api-Key", c.apiKey)

	// More robust WebSocket dialer
	dialer := websocket.Dialer{
		HandshakeTimeout: 15 * time.Second,
		ReadBufferSize:   1024,
		WriteBufferSize:  1024,
	}

	// Attempt connection
	conn, resp, err := dialer.DialContext(ctx, u.String(), headers)
	if err != nil {
		// Detailed error logging
		if resp != nil {
			body, readErr := io.ReadAll(resp.Body)
			log.Printf("WebSocket Connection Error:")
			log.Printf("Status: %s", resp.Status)
			log.Printf("Headers: %+v", resp.Header)
			if readErr == nil {
				log.Printf("Response Body: %s", string(body))
			}
		}
		c.mu.Unlock()
		return fmt.Errorf("websocket connection failed: %w", err)
	}

	// After connection is established
	c.wsConn = conn
	c.mu.Unlock()

	log.Printf("WebSocket connection established successfully")
	handler.OnConnect()

	// Start reading responses
	go c.readResponses(ctx, handler)

	return nil
}

// SendAudioData sends audio data over the WebSocket connection
func (c *Client) SendAudioData(message map[string]interface{}) error {
	c.mu.Lock()
	conn := c.wsConn
	c.mu.Unlock()

	if conn == nil {
		return fmt.Errorf("no active WebSocket connection")
	}

	// Add logging
	msgType, ok := message["type"].(string)
	if ok {
		log.Printf("Sending message type: %s", msgType)
	}

	return conn.WriteJSON(message)
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
			messageType, message, err := c.wsConn.ReadMessage()
			if err != nil {
				log.Printf("Error reading message in Hume client: %v", err) // Add this
				if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseNormalClosure) {
					log.Printf("Unexpected WebSocket close in Hume client: %v", err) // Add this
				}
				handler.OnDisconnect(err)
				return
			}
			// Log raw message
			log.Printf("Received message type: %d, raw message: %d long", messageType, len(message))

			// First check message type
			var typeCheck struct {
				Type string `json:"type"`
			}
			if err := json.Unmarshal(message, &typeCheck); err != nil {
				log.Printf("Error parsing message type: %v", err)
				continue
			}

			var response VoiceChatResponse
			switch typeCheck.Type {
			case "chat_metadata":
				var r ChatMetadata
				if err := json.Unmarshal(message, &r); err != nil {
					log.Printf("Error parsing chat metadata: %v", err)
					continue
				}
				response = r

			case "assistant_message":
				var r AssistantMessage
				if err := json.Unmarshal(message, &r); err != nil {
					log.Printf("Error parsing assistant message: %v", err)
					continue
				}
				response = r

			case "assistant_end":
				var r AssistantEnd
				if err := json.Unmarshal(message, &r); err != nil {
					log.Printf("Error parsing assistant end: %v", err)
					continue
				}
				response = r

			case "audio_output":
				var r AudioResponse
				if err := json.Unmarshal(message, &r); err != nil {
					log.Printf("Error parsing audio response: %v", err)
					continue
				}
				response = r
			}

			if response != nil {
				handler.OnResponse(response)
			}
		}
	}
}

func (c *Client) IsActive() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if !c.isActive || c.wsConn == nil {
		return false
	}

	// Optional: try a ping to verify connection
	err := c.wsConn.WriteControl(websocket.PingMessage, []byte{}, time.Now().Add(time.Second))
	if err != nil {
		log.Printf("Connection check failed: %v", err)
		// Don't modify state here since we only have a read lock
		return false
	}

	return true
}

func (c *Client) ResetState() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	log.Printf("Resetting Hume client state")

	if c.wsConn != nil {
		// Send close message first
		err := c.wsConn.WriteMessage(
			websocket.CloseMessage,
			websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""),
		)
		if err != nil {
			log.Printf("Warning: error sending close message: %v", err)
		}

		// Give a small window for the close message to be sent
		time.Sleep(100 * time.Millisecond)

		// Then close the connection
		err = c.wsConn.Close()
		if err != nil {
			log.Printf("Warning: error closing connection: %v", err)
		}
		c.wsConn = nil
	}

	c.isActive = false
	log.Printf("Hume client state reset completed")
	return nil
}

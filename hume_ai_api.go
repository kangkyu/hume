package hume

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"strconv"
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

type WebsocketMessage struct {
	Type    string          `json:"type"`
	Payload json.RawMessage `json:"payload,omitempty"`
}

func CreateMessage(msgType string, payload interface{}) (WebsocketMessage, error) {
	// Handle different payload types
	var rawPayload json.RawMessage

	switch v := payload.(type) {
	case nil:
		rawPayload = json.RawMessage(`null`)
	case string:
		// Ensure it's a valid JSON string
		rawPayload = json.RawMessage(strconv.Quote(v))
	case []byte:
		// If it's a byte slice, try to parse as JSON
		if json.Valid(v) {
			rawPayload = json.RawMessage(v)
		} else {
			// If not valid JSON, convert to quoted string
			rawPayload = json.RawMessage(strconv.Quote(string(v)))
		}
	default:
		// For other types, marshal to JSON
		jsonData, err := json.Marshal(v)
		if err != nil {
			return WebsocketMessage{}, fmt.Errorf("failed to marshal payload: %w", err)
		}
		rawPayload = json.RawMessage(jsonData)
	}

	return WebsocketMessage{
		Type:    msgType,
		Payload: rawPayload,
	}, nil
}

func ConvertPCMtoWAV(pcmData []byte) []byte {
	var buf bytes.Buffer

	// Calculate sizes
	dataSize := len(pcmData)
	totalSize := dataSize + 44 // 44 bytes for WAV header

	// RIFF chunk descriptor
	buf.WriteString("RIFF")                                      // ChunkID
	binary.Write(&buf, binary.LittleEndian, uint32(totalSize-8)) // ChunkSize
	buf.WriteString("WAVE")                                      // Format

	// fmt sub-chunk
	buf.WriteString("fmt ")                                // Subchunk1ID
	binary.Write(&buf, binary.LittleEndian, uint32(16))    // Subchunk1Size (16 for PCM)
	binary.Write(&buf, binary.LittleEndian, uint16(1))     // AudioFormat (1 for PCM)
	binary.Write(&buf, binary.LittleEndian, uint16(1))     // NumChannels (1 for mono)
	binary.Write(&buf, binary.LittleEndian, uint32(16000)) // SampleRate (16kHz)
	binary.Write(&buf, binary.LittleEndian, uint32(32000)) // ByteRate (SampleRate * NumChannels * BitsPerSample/8)
	binary.Write(&buf, binary.LittleEndian, uint16(2))     // BlockAlign (NumChannels * BitsPerSample/8)
	binary.Write(&buf, binary.LittleEndian, uint16(16))    // BitsPerSample (16 bits)

	// data sub-chunk
	buf.WriteString("data")                                   // Subchunk2ID
	binary.Write(&buf, binary.LittleEndian, uint32(dataSize)) // Subchunk2Size

	// Audio data
	buf.Write(pcmData)

	return buf.Bytes()
}

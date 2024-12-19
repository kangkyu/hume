package hume

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	// "sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

// var upgrader = &websocket.Upgrader{}
var upgrader = &websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

// MockHandler implements VoiceChatHandler for testing
type MockHandler struct {
	mock.Mock
}

func (m *MockHandler) OnConnect() {
	m.Called()
}

func (m *MockHandler) OnDisconnect(err error) {
	m.Called(err)
}

func (m *MockHandler) OnResponse(resp VoiceChatResponse) {
	m.Called(resp)
}

func TestNewClient(t *testing.T) {
	client := NewClient("test-api-key")
	assert.NotNil(t, client)
	assert.Equal(t, "test-api-key", client.apiKey)
	assert.Equal(t, "https://api.hume.ai/v0", client.baseURL)
}

func TestMessageParsing(t *testing.T) {
	tests := []struct {
		name     string
		message  interface{}
		expected VoiceChatResponse
	}{
		{
			name: "ChatMetadata",
			message: ChatMetadata{
				Type:        "chat_metadata",
				ChatGroupID: "group1",
				ChatID:      "chat1",
			},
		},
		{
			name: "AssistantMessage",
			message: AssistantMessage{
				Type: "assistant_message",
				Message: Message{
					Role:    "assistant",
					Content: "Hello",
				},
				FromText: false,
			},
		},
		{
			name: "AudioResponse",
			message: AudioResponse{
				Type:  "audio_output",
				ID:    "audio1",
				Index: 1,
				Data:  "base64-audio-data",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.message)
			assert.NoError(t, err)

			var typeCheck struct {
				Type string `json:"type"`
			}
			err = json.Unmarshal(data, &typeCheck)
			assert.NoError(t, err)

			// Test that the type matches what we expect
			assert.Equal(t, tt.message.(VoiceChatResponse).GetType(), typeCheck.Type)
		})
	}
}

func TestEcho(t *testing.T) {
	// Create test server with the echo handler.
	s := httptest.NewServer(http.HandlerFunc(echo))
	defer s.Close()

	// Convert http://127.0.0.1 to ws://127.0.0.
	u := "ws" + strings.TrimPrefix(s.URL, "http")

	headers := http.Header{}
	headers.Add("X-Hume-Api-Key", "test-api-key")

	// Connect to the server
	ws, _, err := websocket.DefaultDialer.Dial(u, headers)
	if err != nil {
		t.Fatalf("%v", err)
	}
	defer ws.Close()

	// Send message to server, read response and check to see if it's what we expect.
	for i := 0; i < 10; i++ {
		if err := ws.WriteMessage(websocket.TextMessage, []byte("hello")); err != nil {
			t.Fatalf("%v", err)
		}
		_, p, err := ws.ReadMessage()
		if err != nil {
			t.Fatalf("%v", err)
		}
		if string(p) != "hello" {
			t.Fatalf("bad message")
		}
	}
}

func echo(w http.ResponseWriter, r *http.Request) {
	// Verify API key
	if r.Header.Get("X-Hume-Api-Key") != "test-api-key" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	// Upgrade the connection
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		return
	}
	defer conn.Close()

	// Simple echo for testing
	for {
		mt, message, err := conn.ReadMessage()
		if err != nil {
			break
		}
		conn.WriteMessage(mt, message)
	}
}

func TestFullHumeClient(t *testing.T) {
	responses := []string{
		`{
            "type": "chat_metadata",
            "chat_group_id": "test-group-123",
            "chat_id": "test-chat-456"
        }`,
		`{
            "type": "assistant_message",
            "message": {
                "role": "assistant",
                "content": "Hello!"
            },
            "from_text": false
        }`,
		`{
            "type": "audio_output",
            "id": "audio-789",
            "index": 0,
            "data": "test-audio-data",
            "custom_session_id": "test-session"
        }`,
	}

	// Create server that sends these responses
	s := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify API key
		if r.Header.Get("X-Hume-Api-Key") != "test-api-key" {
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}

		// Upgrade to WebSocket
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()

		// Send test responses
		for _, resp := range responses {
			if err := conn.WriteMessage(websocket.TextMessage, []byte(resp)); err != nil {
				return
			}
			time.Sleep(100 * time.Millisecond)
		}

		// Echo any received messages
		for {
			mt, message, err := conn.ReadMessage()
			if err != nil {
				break
			}
			conn.WriteMessage(mt, message)
		}
	}))
	defer s.Close()

	// Create client with handler to collect responses
	var receivedResponses []VoiceChatResponse
	handler := &MockHandler{}

	// Set up all expected mock calls
	handler.On("OnConnect").Return()
	handler.On("OnResponse", mock.Anything).Run(func(args mock.Arguments) {
		resp := args.Get(0).(VoiceChatResponse)
		receivedResponses = append(receivedResponses, resp)
	}).Return()
	handler.On("OnDisconnect", mock.Anything).Return()

	// Create client with TLS config
	client := NewClient("test-api-key", WithTLSConfig(&tls.Config{
		InsecureSkipVerify: true,
	}))
	client.baseURL = strings.Replace(s.URL, "https", "wss", 1)

	// Start voice chat
	ctx := context.Background()
	err := client.StartVoiceChat(ctx, "test-config", handler)
	assert.NoError(t, err)

	// Wait for initial responses
	time.Sleep(500 * time.Millisecond)

	// Verify responses
	assert.Equal(t, len(responses), len(receivedResponses))
	assert.Equal(t, "chat_metadata", receivedResponses[0].GetType())
	assert.Equal(t, "assistant_message", receivedResponses[1].GetType())
	assert.Equal(t, "audio_output", receivedResponses[2].GetType())

	// Send test message
	msg := map[string]interface{}{
		"type": "audio_input",
		"data": "test-input-data",
	}
	err = client.SendAudioData(msg)
	assert.NoError(t, err)

	// Clean close of the connection
	err = client.StopVoiceChat()
	assert.NoError(t, err)

	// Wait for disconnect handler to be called
	time.Sleep(100 * time.Millisecond)

	// Verify all mock expectations were met
	handler.AssertExpectations(t)
}

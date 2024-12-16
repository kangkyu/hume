package hume

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Client handles communication with Hume AI API
type Client struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
}

// ClientOption allows customizing the client
type ClientOption func(*Client)

// NewClient creates a new Hume AI client
func NewClient(apiKey string, opts ...ClientOption) *Client {
	c := &Client{
		apiKey:  apiKey,
		baseURL: "https://api.hume.ai/v1",
		httpClient: &http.Client{
			Timeout: time.Second * 30,
		},
	}

	for _, opt := range opts {
		opt(c)
	}

	return c
}

// WithBaseURL sets a custom base URL
func WithBaseURL(url string) ClientOption {
	return func(c *Client) {
		c.baseURL = url
	}
}

// WithHTTPClient sets a custom HTTP client
func WithHTTPClient(client *http.Client) ClientOption {
	return func(c *Client) {
		c.httpClient = client
	}
}

// VoiceRequest represents the parameters for a voice analysis request
type VoiceRequest struct {
	Audio     io.Reader
	ModelName string // Optional, defaults to latest
}

// VoiceResponse represents the response from voice analysis
type VoiceResponse struct {
	Text       string                 `json:"text"`
	Emotions   map[string]float64     `json:"emotions"`
	Transcript string                 `json:"transcript"`
	Meta       map[string]interface{} `json:"meta"`
}

// ProcessVoice analyzes voice data and returns insights
func (c *Client) ProcessVoice(ctx context.Context, req VoiceRequest) (*VoiceResponse, error) {
	endpoint := fmt.Sprintf("%s/voice/analyze", c.baseURL)

	// Read audio data
	audioData, err := io.ReadAll(req.Audio)
	if err != nil {
		return nil, fmt.Errorf("reading audio data: %w", err)
	}

	// Create request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", endpoint, bytes.NewReader(audioData))
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}

	// Set headers
	httpReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.apiKey))
	httpReq.Header.Set("Content-Type", "audio/wav") // Adjust based on actual audio format

	if req.ModelName != "" {
		httpReq.Header.Set("X-Model-Name", req.ModelName)
	}

	// Make request
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("making request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error: status=%d body=%s", resp.StatusCode, string(body))
	}

	// Parse response
	var result VoiceResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decoding response: %w", err)
	}

	return &result, nil
}

// StreamConfig represents configuration for streaming voice analysis
type StreamConfig struct {
	SampleRate  int
	NumChannels int
	ModelName   string
}

// StreamHandler processes streaming voice data
type StreamHandler interface {
	OnResult(VoiceResponse)
	OnError(error)
}

// StreamVoice processes streaming voice data in real-time
func (c *Client) StreamVoice(ctx context.Context, config StreamConfig, handler StreamHandler) error {
	// Implement WebSocket-based streaming here
	// This would connect to Hume's streaming endpoint
	// and handle real-time voice processing
	return fmt.Errorf("streaming not yet implemented")
}

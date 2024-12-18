package hume

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"strconv"
)

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

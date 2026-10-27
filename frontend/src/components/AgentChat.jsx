import React, { useState } from "react";
import { Brain, Send, Sparkles, AlertCircle, PlayCircle, CornerDownRight } from "lucide-react";
import { queryAgent } from "../services/api";

export default function AgentChat({ kpis, onTriggerAction }) {
  const [messages, setMessages] = useState([
    {
      sender: "agent",
      text: `### 🤖 OptiChain Autonomous Copilot
Welcome to the OptiChain AI decision intelligence room. I continuously monitor the digital twin environment, analyze lead-time volatility, and stress-test inventory policies.

**Current Health Check:**
- Service Level: **92.4% OTIF**
- Active Risk: Supplier lead-time stochasticity at SUP_01
- Recommendation: Consider running a Constrained MILP optimization cycle.

How may I assist your supply chain decisions today?`,
      suggested_actions: [
        "Why did stockouts occur between Day 45-55?",
        "Compare Constrained MILP vs Deep RL (PPO)",
        "Recommend optimal safety stock buffer",
      ],
    },
  ]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);

  const handleSend = async (textToSend) => {
    const query = textToSend || input;
    if (!query.trim()) return;

    const userMsg = { sender: "user", text: query };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsTyping(true);

    try {
      const res = await queryAgent(query, kpis);
      const agentMsg = {
        sender: "agent",
        text: res.response,
        suggested_actions: res.suggested_actions || [],
      };
      setMessages((prev) => [...prev, agentMsg]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          sender: "agent",
          text: "I encountered an error querying the diagnostic engine. Please retry.",
        },
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className="agent-chat-page">
      <div className="glass-panel agent-chat-card">
        {/* Chat Header */}
        <div className="agent-chat-header">
          <div className="agent-avatar">
            <Brain size={22} className="brain-icon" />
          </div>
          <div>
            <div className="agent-name">OptiChain Reasoning Agent</div>
            <div className="agent-role">Digital Twin Autonomous Decision Copilot</div>
          </div>
          <div className="agent-status-badge">
            <span className="dot pulse"></span>
            <span>Reasoning Active</span>
          </div>
        </div>

        {/* Message Stream */}
        <div className="agent-messages-stream">
          {messages.map((m, idx) => (
            <div key={idx} className={`message-row ${m.sender}`}>
              <div className="message-bubble">
                <div
                  className="message-markdown"
                  dangerouslySetInnerHTML={{
                    __html: m.text
                      .replace(/### (.*)/g, "<h4>$1</h4>")
                      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
                      .replace(/\*(.*?)\*/g, "<em>$1</em>")
                      .replace(/- (.*)/g, "<li>$1</li>")
                      .replace(/\n\n/g, "<br/><br/>"),
                  }}
                />

                {m.suggested_actions && m.suggested_actions.length > 0 && (
                  <div className="suggested-chips">
                    <span className="chips-label">Suggested Inquiries:</span>
                    {m.suggested_actions.map((act, i) => (
                      <button
                        key={i}
                        onClick={() => handleSend(act)}
                        className="chip-btn"
                      >
                        <CornerDownRight size={12} />
                        <span>{act}</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}

          {isTyping && (
            <div className="message-row agent">
              <div className="message-bubble typing-bubble">
                <Sparkles size={16} className="spin" />
                <span>Synthesizing supply chain telemetry...</span>
              </div>
            </div>
          )}
        </div>

        {/* Input Bar */}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSend();
          }}
          className="agent-input-form"
        >
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about disruptions, policy comparisons, or safety stock recommendations..."
            className="agent-text-input"
          />
          <button type="submit" disabled={!input.trim() || isTyping} className="btn-send">
            <Send size={16} />
          </button>
        </form>
      </div>
    </div>
  );
}

import React, { useState } from "react";
import DOMPurify from "dompurify";
import { marked } from "marked";

const ChatInterface = ({ chat }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  const send = async () => {
    if (!input) return;
    const userMsg = { sender: "user", text: input };
    setMessages((prev) => [...prev, userMsg]);

    const reply = await chat.send({ text: input });
    const botMsg = { sender: "bot", text: reply.response };
    setMessages((prev) => [...prev, botMsg]);

    setInput("");
  };

  return (
    <div className="w-full max-w-2xl bg-black border border-red-700 p-4 rounded-lg shadow-lg shadow-red-500/30">
      <div className="h-96 overflow-y-auto mb-4 space-y-2">
        {messages.map((m, i) => (
          <div
            key={i}
            className={`p-2 rounded ${m.sender === "user" ? "bg-red-900/30" : "bg-cyan-900/20"}`}
            dangerouslySetInnerHTML={{
              __html: DOMPurify.sanitize(marked(m.text || ""))
            }}
          />
        ))}
      </div>

      <div className="flex gap-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          className="flex-grow p-2 bg-black border border-red-600 rounded"
          placeholder="Type your command…"
        />
        <button 
          onClick={send}
          className="bg-red-600 px-4 py-2 rounded hover:bg-red-700"
        >
          Send
        </button>
      </div>
    </div>
  );
};

export default ChatInterface;

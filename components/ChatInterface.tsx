// src/components/ChatInterface.tsx
import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import DOMPurify from "dompurify";
import { AI_BACKEND } from "../constants";

type Props = {
  isAdmin?: boolean;
};

const ChatInterface: React.FC<Props> = ({ isAdmin = false }) => {
  const [messages, setMessages] = useState<{ role: "user" | "assistant"; text: string }[]>([]);
  const [input, setInput] = useState("");
  const [thinking, setThinking] = useState(false);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const send = async () => {
    if (!input.trim()) return;
    const text = input.trim();
    const userMsg = { role: "user", text };
    setMessages((m) => [...m, userMsg]);
    setInput("");
    setThinking(true);

    try {
      const r = await axios.post(`${AI_BACKEND}/chat`, { prompt: text }, { timeout: 120000 });
      const reply = r.data?.response ?? JSON.stringify(r.data);
      const botMsg = { role: "assistant", text: reply };
      setMessages((m) => [...m, botMsg]);
    } catch (err: any) {
      setMessages((m) => [...m, { role: "assistant", text: "⚠️ Error: failed to connect to local model." }]);
    } finally {
      setThinking(false);
    }
  };

  const genImage = async () => {
    if (!input.trim()) return;
    const prompt = input.trim();
    setThinking(true);
    try {
      const r = await axios.post(`${AI_BACKEND}/image`, { prompt, width: 512, height: 512 }, { timeout: 300000 });
      const b64 = r.data?.image_base64 ?? r.data?.image;
      if (b64) {
        const imgData = "data:image/png;base64," + b64;
        setMessages((m) => [...m, { role: "assistant", text: imgData }]);
      } else {
        setMessages((m) => [...m, { role: "assistant", text: "⚠️ Image generation returned no image." }]);
      }
    } catch {
      setMessages((m) => [...m, { role: "assistant", text: "⚠️ Error generating image." }]);
    } finally {
      setThinking(false);
    }
  };

  const renderMessage = (m: { role: string; text: string }, i: number) => {
    if (m.text.startsWith("data:image/")) {
      // image
      return <img key={i} src={m.text} alt="generated" className="max-w-full rounded" />;
    }
    return (
      <div
        key={i}
        className={`p-2 rounded ${m.role === "user" ? "bg-red-900/30" : "bg-cyan-900/10"}`}
        dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(m.text) }}
      />
    );
  };

  return (
    <div className="w-full max-w-3xl p-4 bg-gray-900 border border-red-700 rounded-lg">
      <div className="flex justify-between items-center mb-3">
        <h3 className="font-bold text-lg">WormGPT — Local</h3>
        <div className="text-sm text-gray-400">{isAdmin ? "Admin" : "User"}</div>
      </div>

      <div ref={scrollRef} className="h-96 overflow-y-auto bg-black p-3 mb-3 space-y-2">
        {messages.map((m, i) => renderMessage(m, i))}
        {thinking && <div className="text-sm text-gray-400">🤖 Model thinking...</div>}
      </div>

      <div className="flex gap-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
          placeholder="Type prompt or image description..."
          className="flex-1 p-2 bg-black border border-gray-700 rounded"
        />
        <button onClick={send} className="bg-red-600 px-4 py-2 rounded">
          Send
        </button>
        <button onClick={genImage} className="bg-cyan-600 px-3 py-2 rounded">
          Image
        </button>
      </div>
    </div>
  );
};

export default ChatInterface;

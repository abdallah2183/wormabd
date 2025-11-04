// src/components/GateScreen.tsx
import React, { useState } from "react";
import { ADMIN_CODE, TEMP_CODE, TEMP_DURATION_MS } from "../constants";

type Props = {
  onAuthSuccess: (admin?: boolean) => void;
};

const GateScreen: React.FC<Props> = ({ onAuthSuccess }) => {
  const [code, setCode] = useState("");
  const [error, setError] = useState<string | null>(null);

  const submit = () => {
    setError(null);
    const trimmed = code.trim();
    if (!trimmed) {
      setError("Enter a code.");
      return;
    }

    if (trimmed === ADMIN_CODE) {
      // admin permanent
      onAuthSuccess(true);
      return;
    }

    if (trimmed === TEMP_CODE) {
      // temporary: set expiry for 1 hour
      const expiry = Date.now() + TEMP_DURATION_MS;
      localStorage.setItem("wormGPT_access_expiry", expiry.toString());
      onAuthSuccess(false);
      return;
    }

    setError("Invalid code.");
  };

  return (
    <div className="w-full max-w-md bg-gray-900 border border-red-700 p-6 rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-4 text-gradient-crimson">WormGPT Access</h2>
      <p className="mb-4 text-sm text-gray-400">Enter access code to continue.</p>

      <input
        value={code}
        onChange={(e) => setCode(e.target.value)}
        placeholder="Enter access code"
        className="w-full p-2 mb-3 bg-black border border-gray-700 rounded"
      />
      <div className="flex gap-2">
        <button onClick={submit} className="px-4 py-2 bg-red-600 rounded hover:bg-red-700">
          Unlock
        </button>
      </div>
      {error && <p className="mt-3 text-red-400">{error}</p>}
      <p className="mt-4 text-xs text-gray-500">Temporary code grants 1 hour access. Admin code is permanent.</p>
    </div>
  );
};

export default GateScreen;

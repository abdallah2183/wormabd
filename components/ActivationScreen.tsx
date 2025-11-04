// src/components/ActivationScreen.tsx
import React, { useState } from "react";

type Props = {
  onActivationComplete: () => void;
};

const ActivationScreen: React.FC<Props> = ({ onActivationComplete }) => {
  const [agree, setAgree] = useState(false);

  const activate = () => {
    if (!agree) return;
    onActivationComplete();
  };

  return (
    <div className="w-full max-w-2xl bg-gray-900 border border-red-700 p-6 rounded-lg shadow-lg text-center">
      <h2 className="text-2xl font-bold mb-4 text-gradient-crimson">Activation</h2>
      <p className="mb-4 text-gray-400">Activate WormGPT local core. This device will run the models locally.</p>

      <label className="flex items-center gap-2 justify-center mb-4">
        <input type="checkbox" checked={agree} onChange={() => setAgree(!agree)} />
        <span className="text-sm text-gray-300">I understand that models run locally on this device.</span>
      </label>

      <div>
        <button onClick={activate} className="px-6 py-2 bg-red-600 rounded disabled:opacity-50" disabled={!agree}>
          Activate
        </button>
      </div>
    </div>
  );
};

export default ActivationScreen;

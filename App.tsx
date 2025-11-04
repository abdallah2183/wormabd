// src/App.tsx
import React, { useState, useEffect } from "react";
import GateScreen from "./components/GateScreen";
import ActivationScreen from "./components/ActivationScreen";
import ChatInterface from "./components/ChatInterface";
import { ADMIN_CODE } from "./constants";

const App: React.FC = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isActivated, setIsActivated] = useState(false);
  const [isAdmin, setIsAdmin] = useState(false);

  useEffect(() => {
    const admin = localStorage.getItem("wormGPT_isAdmin");
    if (admin === "true") {
      setIsAuthenticated(true);
      setIsAdmin(true);
      return;
    }

    const expiry = localStorage.getItem("wormGPT_access_expiry");
    if (expiry && Date.now() < parseInt(expiry)) {
      setIsAuthenticated(true);
    }
  }, []);

  const handleAuthSuccess = (adminFlag = false) => {
    setIsAuthenticated(true);
    if (adminFlag) {
      setIsAdmin(true);
      localStorage.setItem("wormGPT_isAdmin", "true");
    }
  };

  const handleActivationComplete = () => {
    setIsActivated(true);
  };

  return (
    <div className="min-h-screen bg-black text-gray-200 font-mono flex items-center justify-center p-4">
      {!isAuthenticated ? (
        <GateScreen onAuthSuccess={handleAuthSuccess} />
      ) : !isActivated ? (
        <ActivationScreen onActivationComplete={handleActivationComplete} />
      ) : (
        <ChatInterface isAdmin={isAdmin} />
      )}
    </div>
  );
};

export default App;

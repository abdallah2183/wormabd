import React, { useEffect, useRef } from 'react';
import { ChatMessage } from '../types';
import ObsidianIcon from './ObsidianIcon';

declare var marked: any;

interface MessageProps {
  message: ChatMessage;
  isLoading?: boolean;
}

const Message: React.FC<MessageProps> = ({ message, isLoading = false }) => {
  const isUser = message.role === 'user';
  const contentRef = useRef<HTMLDivElement>(null);

  const clipboardIcon = `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-4 h-4"><path stroke-linecap="round" stroke-linejoin="round" d="M15.666 3.888A2.25 2.25 0 0 0 13.5 2.25h-3c-1.03 0-1.9.693-2.166 1.638m7.332 0c.055.194.084.4.084.612v3.042m-7.416 0v3.042c0 .212.03.418.084.612m7.332 0c.646.049 1.288.11 1.927.184 1.1.128 1.907 1.077 1.907 2.185V19.5a2.25 2.25 0 0 1-2.25 2.25H6.75A2.25 2.25 0 0 1 4.5 19.5V6.257c0-1.108.806-2.057 1.907-2.185a48.208 48.208 0 0 1 1.927-.184" /></svg>`;
  const checkIcon = `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-4 h-4"><path stroke-linecap="round" stroke-linejoin="round" d="m4.5 12.75 6 6 9-13.5" /></svg>`;

  useEffect(() => {
    if (contentRef.current) {
      const preElements = contentRef.current.querySelectorAll('pre');
      preElements.forEach((pre) => {
        if (pre.querySelector('.code-header')) return; // Already processed

        const codeEl = pre.querySelector('code');
        const lang = codeEl?.className.match(/language-(\S+)/)?.[1] || 'shell';
        
        pre.style.position = 'relative';

        const header = document.createElement('div');
        header.className = 'code-header flex justify-between items-center bg-gray-900/70 text-gray-400 text-xs px-3 py-1.5 rounded-t-md border-b border-black/30';
        
        const langSpan = document.createElement('span');
        langSpan.innerText = lang;
        
        const button = document.createElement('button');
        button.innerHTML = clipboardIcon + ' Copy';
        button.className = 'flex items-center gap-1.5 p-1.5 -my-1.5 -mr-1.5 text-gray-400 rounded-md hover:bg-gray-700/50 hover:text-gray-200 transition-colors duration-200';
        button.setAttribute('aria-label', 'Copy code to clipboard');
        
        button.onclick = () => {
          const code = pre.querySelector('code')?.innerText || '';
          navigator.clipboard.writeText(code).then(() => {
            button.innerHTML = checkIcon + ' Copied';
            button.classList.add('text-green-400');
            setTimeout(() => {
              button.innerHTML = clipboardIcon + ' Copy';
              button.classList.remove('text-green-400');
            }, 2000);
          });
        };
        
        header.appendChild(langSpan);
        header.appendChild(button);
        pre.insertBefore(header, pre.firstChild);
        codeEl?.classList.add('rounded-b-md', 'mt-0');
      });
    }
  }, [message.content, isLoading]);

  const parsedHTML = isUser || typeof marked === 'undefined' 
    ? message.content.replace(/<pre>/g, '<pre class="hljs">')
    : marked.parse(message.content || '');


  if (isUser) {
    return (
      <div className="flex justify-end ml-10">
        <div className="max-w-xl lg:max-w-3xl bg-cyan-900/20 text-cyan-200 p-3 rounded-lg border border-cyan-700/50 clip-corner shadow-[0_0_15px_var(--color-cyan-glow)]">
           <p className="font-semibold text-cyan-400 text-sm mb-2 opacity-80">:: User Command</p>
           {message.imageUrl && (
            <img 
                src={message.imageUrl} 
                alt="User upload" 
                className="max-w-xs rounded-md border-2 border-cyan-700/50 mb-2" 
            />
           )}
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-start space-x-4 mr-10">
      <div className="flex-shrink-0 pt-1">
          <ObsidianIcon />
      </div>
      <div className="flex-1 min-w-0 bg-black/30 p-4 rounded-lg border border-red-900/50 clip-corner shadow-[0_0_20px_var(--color-crimson-glow)]">
        {isLoading && !message.content ? (
            <div className="w-3 h-5 bg-red-500 animate-pulse" />
        ) : (
          <div className="prose prose-invert prose-pre:bg-[#1a1d23] prose-pre:p-0 prose-pre:rounded-md" dangerouslySetInnerHTML={{ __html: parsedHTML }} ref={contentRef} />
        )}
      </div>
    </div>
  );
};

export default Message;
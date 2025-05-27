class ChatManager {
    constructor() {
        this.currentChatId = null;
        this.chats = [];
        this.department = '';
        this.semester = '';
        this.subject = '';
    }
    
    async init(department, semester, subject) {
        this.department = department;
        this.semester = semester;
        this.subject = subject;
        
        // Load initial chat history
        await this.loadChatHistory();
        
        // Check for chat_id in URL
        const urlParams = new URLSearchParams(window.location.search);
        const chatId = urlParams.get('chat_id');
        if (chatId) {
            await this.loadChat(chatId);
        }
        
        // Set up event listeners
        this.setupEventListeners();
    }
    
    async loadChatHistory() {
        try {
            const response = await fetch(`${API_BASE_URL}/chats?department=${encodeURIComponent(this.department)}&semester=${encodeURIComponent(this.semester)}&subject=${encodeURIComponent(this.subject)}`);
            this.chats = await response.json();
            this.renderChatHistory();
        } catch (error) {
            console.error('Error loading chat history:', error);
        }
    }
    
    renderChatHistory(filter = '') {
        const historyList = document.getElementById('history-list');
        
        // Filter chats
        const filteredChats = this.chats.filter(chat => 
            chat.title.toLowerCase().includes(filter.toLowerCase()) ||
            chat.messages.some(msg => msg.content.toLowerCase().includes(filter.toLowerCase()))
            .sort((a, b) => b.updated_at - a.updated_at));
        
        // Clear existing items
        historyList.innerHTML = '';
        
        if (filteredChats.length === 0) {
            historyList.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-comment-slash"></i>
                    <p>${filter ? 'No matching chats' : 'No recent chats'}</p>
                </div>
            `;
            return;
        }
        
        // Add each chat to the list
        filteredChats.forEach(chat => {
            const lastMessage = chat.messages[chat.messages.length - 1]?.content || '';
            const preview = lastMessage.length > 50 ? lastMessage.substring(0, 50) + '...' : lastMessage;
            const date = new Date(chat.updated_at * 1000);
            const timeString = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            
            const chatElement = document.createElement('button');
            chatElement.className = `history-item ${this.currentChatId === chat.id ? 'active' : ''}`;
            chatElement.dataset.chatId = chat.id;
            chatElement.innerHTML = `
                <div class="history-item-content">
                    <div class="history-item-title">${chat.title}</div>
                    <div class="history-item-preview">${preview}</div>
                    <div class="history-item-time">${timeString}</div>
                </div>
                <div class="history-item-actions">
                    <button class="history-item-action delete-chat-btn" title="Delete chat">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            `;
            
            // Add click handler
            chatElement.addEventListener('click', (e) => {
                if (!e.target.closest('.history-item-action')) {
                    this.loadChat(chat.id);
                }
            });
            
            // Add delete handler
            const deleteBtn = chatElement.querySelector('.delete-chat-btn');
            deleteBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.deleteChat(chat.id);
            });
            
            historyList.appendChild(chatElement);
        });
    }
    
    async loadChat(chatId) {
        const chatElement = document.querySelector(`.history-item[data-chat-id="${chatId}"]`);
        
        try {
            // Show loading state
            if (chatElement) {
                chatElement.classList.add('loading');
                chatElement.querySelector('.history-item-content').innerHTML = '<div class="loading-spinner"></div>';
            }
            
            // Fetch the chat
            const response = await fetch(`${API_BASE_URL}/chats/${chatId}`);
            const chat = await response.json();
            
            // Clear current chat
            document.getElementById('chat-messages').innerHTML = '';
            
            // Set as current chat
            this.currentChatId = chatId;
            
            // Render all messages
            chat.messages.forEach(msg => {
                addMessage(msg.role === 'user' ? 'user' : 'ai', msg.content, true);
            });
            
            // Update URL
            const url = new URL(window.location);
            url.searchParams.set('chat_id', chatId);
            window.history.pushState({}, '', url);
            
            // Update UI
            this.renderChatHistory();
            document.getElementById('welcome-container')?.remove();
            
            // Scroll to bottom
            const chatMessages = document.getElementById('chat-messages');
            chatMessages.scrollTop = chatMessages.scrollHeight;
            
        } catch (error) {
            console.error('Error loading chat:', error);
            alert('Failed to load chat. Please try again.');
        } finally {
            // Remove loading state
            if (chatElement) {
                chatElement.classList.remove('loading');
                this.renderChatHistory();
            }
        }
    }
    
    async saveCurrentChat(messages) {
        if (messages.length === 0) return null;
        
        try {
            // Generate title from first message
            const title = messages[0].content.substring(0, 50) + (messages[0].content.length > 50 ? '...' : '');
            
            let chat;
            
            if (this.currentChatId) {
                // Update existing chat
                const response = await fetch(`${API_BASE_URL}/chats/${this.currentChatId}`, {
                    method: 'PUT',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ messages })
                });
                chat = await response.json();
            } else {
                // Create new chat
                const response = await fetch(`${API_BASE_URL}/chats`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        department: this.department,
                        semester: this.semester,
                        subject: this.subject,
                        title,
                        messages
                    })
                });
                chat = await response.json();
                this.currentChatId = chat.id;
            }
            
            // Refresh history
            await this.loadChatHistory();
            
            return chat;
        } catch (error) {
            console.error('Error saving chat:', error);
            return null;
        }
    }
    
    async deleteChat(chatId) {
        if (!confirm('Are you sure you want to delete this chat?')) return;
        
        try {
            await fetch(`${API_BASE_URL}/chats/${chatId}`, {
                method: 'DELETE'
            });
            
            // If deleting current chat, clear the view
            if (this.currentChatId === chatId) {
                this.currentChatId = null;
                document.getElementById('chat-messages').innerHTML = '';
                addWelcomeMessage();
                
                // Update URL
                const url = new URL(window.location);
                url.searchParams.delete('chat_id');
                window.history.pushState({}, '', url);
            }
            
            // Refresh history
            await this.loadChatHistory();
        } catch (error) {
            console.error('Error deleting chat:', error);
            alert('Failed to delete chat. Please try again.');
        }
    }
    
    async startNewChat() {
        // Save current chat if it has messages
        const messages = this.getCurrentMessages();
        if (messages.length > 0) {
            await this.saveCurrentChat(messages);
        }
        
        // Reset to new chat
        this.currentChatId = null;
        document.getElementById('chat-messages').innerHTML = '';
        addWelcomeMessage();
        
        // Update URL
        const url = new URL(window.location);
        url.searchParams.delete('chat_id');
        window.history.pushState({}, '', url);
    }
    
    getCurrentMessages() {
        const messages = [];
        const messageElements = document.querySelectorAll('.message');
        
        messageElements.forEach(el => {
            if (el.classList.contains('message-user')) {
                const content = el.querySelector('.message-bubble').textContent;
                messages.push({ role: 'user', content });
            } else if (el.classList.contains('message-ai') && el.id !== 'typing-indicator') {
                const content = el.querySelector('.message-bubble').textContent;
                messages.push({ role: 'assistant', content });
            }
        });
        
        return messages;
    }
    
    setupEventListeners() {
        // New chat button
        document.getElementById('new-chat-btn').addEventListener('click', () => {
            this.startNewChat();
        });
        
        // Search functionality
        const searchBtn = document.getElementById('search-chats-btn');
        const searchContainer = document.getElementById('search-container');
        const searchInput = document.getElementById('chat-search');
        const clearSearchBtn = document.getElementById('clear-search-btn');
        
        searchBtn.addEventListener('click', () => {
            searchContainer.classList.toggle('visible');
            if (searchContainer.classList.contains('visible')) {
                searchInput.focus();
            }
        });
        
        searchInput.addEventListener('input', () => {
            this.renderChatHistory(searchInput.value.trim());
        });
        
        clearSearchBtn.addEventListener('click', () => {
            searchInput.value = '';
            this.renderChatHistory('');
        });
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Get parameters from URL
    const urlParams = new URLSearchParams(window.location.search);
    const department = urlParams.get('department');
    const semester = urlParams.get('semester');
    const subject = urlParams.get('subject');
    
    // Validate parameters
    if (!department || !semester || !subject) {
        window.location.href = 'index.html';
        return;
    }
    
    // Initialize chat manager
    window.chatManager = new ChatManager();
    chatManager.init(department, semester, subject);
    
    // Update current selection display
    document.getElementById('current-dept').textContent = department;
    document.getElementById('current-sem').textContent = semester;
    document.getElementById('current-subj').textContent = subject;
    document.getElementById('current-subject').textContent = subject;
});
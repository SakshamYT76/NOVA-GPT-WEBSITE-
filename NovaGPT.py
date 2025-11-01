# novagpt_complete.py
"""
NovaGPT - Your All-in-One AI Universe
Complete Full-Stack Python AI Assistant
"""

import os
import asyncio
import json
import base64
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from enum import Enum
import secrets

# Third-party imports
try:
    from fastapi import FastAPI, HTTPException, Depends, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import HTTPBearer
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, FileResponse
    import uvicorn
    from pydantic import BaseModel, EmailStr
    import jwt
    from bcrypt import hashpw, gensalt, checkpw
    import requests
    import openai
    import anthropic
    import replicate
    import cloudinary
    import cloudinary.uploader
    import streamlit as st
    import plotly.graph_objects as go
    from streamlit_option_menu import option_menu
    import markdown
    import pygments
    from pygments.lexers import get_lexer_by_name
    from pygments.formatters import HtmlFormatter
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Please install: pip install fastapi uvicorn pydantic pyjwt bcrypt requests openai anthropic replicate cloudinary streamlit plotly streamlit-option-menu markdown pygments")
    exit(1)

# Configuration
class Config:
    APP_NAME = "NovaGPT"
    VERSION = "1.0.0"
    DEBUG = True
    
    # Security
    SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-key-change-in-production")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    
    # Database (using in-memory for demo)
    DATABASE_URL = "memory"
    
    # APIs (you should set these in environment variables)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-key")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "your-anthropic-key")
    REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "your-replicate-token")
    SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "your-serpapi-key")
    CLOUDINARY_URL = os.getenv("CLOUDINARY_URL", "your-cloudinary-url")
    
    # Initialize AI clients
    openai.api_key = OPENAI_API_KEY
    anthropic_client = anthropic.Client(api_key=ANTHROPIC_API_KEY)
    cloudinary.config(cloudinary_url=CLOUDINARY_URL)

# Enums and Models
class ToolType(str, Enum):
    CHAT = "chat"
    SEARCH = "search"
    IMAGE = "image"
    STUDY = "study"
    HOMEWORK = "homework"
    PROMPT = "prompt"

class UserRole(str, Enum):
    FREE = "free"
    PRO = "pro"
    PREMIUM = "premium"

# Pydantic Models
class Source(BaseModel):
    url: str
    title: str
    domain: str

class Message(BaseModel):
    role: str
    content: str
    images: List[str] = []
    sources: List[Source] = []
    timestamp: datetime

class UserBase(BaseModel):
    name: str
    email: EmailStr
    phone: Optional[str] = None
    image: Optional[str] = None
    role: UserRole = UserRole.FREE
    credits: int = 100

class UserCreate(UserBase):
    password: Optional[str] = None
    google_id: Optional[str] = None

class User(UserBase):
    id: str
    created_at: datetime
    updated_at: datetime

class Chat(BaseModel):
    id: str
    user_id: str
    title: str
    messages: List[Message]
    tool: ToolType
    created_at: datetime
    updated_at: datetime

class ChatCreate(BaseModel):
    title: str
    tool: ToolType = ToolType.CHAT

class MessageCreate(BaseModel):
    content: str
    tool: ToolType = ToolType.CHAT
    images: List[str] = []

class ImageGenerationOptions(BaseModel):
    prompt: str
    style: str = "realistic"
    size: str = "1024x1024"
    quality: str = "standard"

class AuthResponse(BaseModel):
    access_token: str
    token_type: str
    user: User

class TokenData(BaseModel):
    email: Optional[str] = None

# Authentication and Security
class AuthService:
    security = HTTPBearer()
    
    @staticmethod
    def verify_password(plain_password, hashed_password):
        return checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    
    @staticmethod
    def get_password_hash(password):
        return hashpw(password.encode('utf-8'), gensalt()).decode('utf-8')
    
    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=Config.ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, Config.SECRET_KEY, algorithm=Config.ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def verify_token(token: str):
        try:
            payload = jwt.decode(token, Config.SECRET_KEY, algorithms=[Config.ALGORITHM])
            email: str = payload.get("sub")
            if email is None:
                return None
            return TokenData(email=email)
        except jwt.PyJWTError:
            return None

# Database (In-Memory for Demo)
class Database:
    def __init__(self):
        self.users = {}
        self.chats = {}
        self.sessions = {}
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        return self.users.get(email)
    
    async def create_user(self, user_data: dict) -> User:
        user_id = secrets.token_hex(16)
        user_data["id"] = user_id
        user_data["created_at"] = datetime.utcnow()
        user_data["updated_at"] = datetime.utcnow()
        
        user = User(**user_data)
        self.users[user_data["email"]] = user
        return user
    
    async def get_user_chats(self, user_id: str) -> List[Chat]:
        return [chat for chat in self.chats.values() if chat.user_id == user_id]
    
    async def create_chat(self, chat_data: dict) -> Chat:
        chat_id = secrets.token_hex(16)
        chat_data["id"] = chat_id
        chat_data["created_at"] = datetime.utcnow()
        chat_data["updated_at"] = datetime.utcnow()
        
        chat = Chat(**chat_data)
        self.chats[chat_id] = chat
        return chat
    
    async def update_chat(self, chat_id: str, update_data: dict):
        if chat_id in self.chats:
            chat = self.chats[chat_id]
            for key, value in update_data.items():
                setattr(chat, key, value)
            chat.updated_at = datetime.utcnow()

# AI Services
class AIService:
    @staticmethod
    async def chat_completion(messages: List[Dict[str, str]], tool: str = "chat") -> Dict[str, Any]:
        try:
            if tool in ["study", "homework"]:
                # Use Claude for educational content
                response = Config.anthropic_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=4000,
                    messages=messages
                )
                return {
                    "content": response.content[0].text,
                    "sources": []
                }
            else:
                # Use GPT-4 for general chat
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=messages,
                    max_tokens=2000
                )
                return {
                    "content": response.choices[0].message.content,
                    "sources": []
                }
        except Exception as e:
            return {
                "content": f"I apologize, but I encountered an error: {str(e)}",
                "sources": []
            }

    @staticmethod
    async def web_search(query: str) -> Dict[str, Any]:
        try:
            # Mock search implementation - replace with actual SerpAPI
            mock_results = [
                {
                    "title": f"Search Result for: {query}",
                    "link": "https://example.com/result1",
                    "snippet": f"This is a mock search result for {query}. In production, this would use SerpAPI."
                },
                {
                    "title": f"Another result for: {query}",
                    "link": "https://example.com/result2", 
                    "snippet": f"Another mock result showing how search would work for {query}."
                }
            ]
            
            sources = []
            summary_content = f"Based on my search for '{query}', here's what I found:\n\n"
            
            for i, result in enumerate(mock_results, 1):
                sources.append(Source(
                    url=result.get("link"),
                    title=result.get("title"),
                    domain="example.com"
                ))
                summary_content += f"{i}. [{result.get('title')}]({result.get('link')})\n"
                summary_content += f"   {result.get('snippet', '')}\n\n"
            
            summary_content += "\n**Sources:**\n" + "\n".join(
                [f"- [{source.title}]({source.url})" for source in sources]
            )
            
            return {
                "content": summary_content,
                "sources": sources
            }
            
        except Exception as e:
            return {
                "content": f"I encountered an error while searching: {str(e)}",
                "sources": []
            }

    @staticmethod
    async def generate_image(options: ImageGenerationOptions) -> Dict[str, Any]:
        try:
            # Mock image generation - replace with actual DALL-E
            mock_image_url = "https://via.placeholder.com/1024x1024/667eea/ffffff?text=AI+Generated+Image"
            
            # In production, use:
            # response = openai.Image.create(
            #     prompt=options.prompt,
            #     n=1,
            #     size=options.size
            # )
            # image_url = response.data[0].url
            
            return {
                "success": True,
                "image_url": mock_image_url,
                "prompt": options.prompt
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    @staticmethod
    async def study_helper(question: str, subject: str = "general") -> Dict[str, Any]:
        system_prompt = f"""You are an expert {subject} tutor. Explain the concept in simple, step-by-step terms. 
        Use analogies and examples to make it understandable. Break down complex ideas into smaller parts.
        Always end with a summary and practical applications."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        return await AIService.chat_completion(messages, "study")

    @staticmethod
    async def homework_solver(problem: str, show_steps: bool = True) -> Dict[str, Any]:
        system_prompt = """You are a homework helper. Provide detailed, step-by-step solutions. 
        Explain the reasoning behind each step. Include relevant formulas and concepts.
        Make sure the solution is educational and helps understanding."""
        
        if show_steps:
            problem += " Please show all steps and explain your reasoning."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem}
        ]
        
        return await AIService.chat_completion(messages, "homework")

    @staticmethod
    async def prompt_generator(user_prompt: str, category: str = "general") -> Dict[str, Any]:
        categories = {
            "writing": "writing assistant prompts",
            "coding": "programming and code generation", 
            "art": "AI art and image generation",
            "study": "educational and learning",
            "productivity": "productivity and business"
        }
        
        system_prompt = f"""You are an expert prompt engineer. Generate optimized AI prompts for {categories.get(category, 'general use')}. 
        Create 3 variations of the user's prompt: basic, detailed, and expert level.
        For each variation, explain why it's effective and what makes it work well."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Original prompt: {user_prompt}"}
        ]
        
        return await AIService.chat_completion(messages, "prompt")

# FastAPI Backend Application
class NovaGPTBackend:
    def __init__(self):
        self.app = FastAPI(title=Config.APP_NAME, version=Config.VERSION)
        self.db = Database()
        self.auth = AuthService()
        self.setup_middleware()
        self.setup_routes()
    
    def setup_middleware(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        # Root endpoint
        @self.app.get("/")
        async def root():
            return {"message": f"Welcome to {Config.APP_NAME}", "version": Config.VERSION}
        
        # Authentication endpoints
        @self.app.post("/auth/register", response_model=AuthResponse)
        async def register(user_data: UserCreate):
            existing_user = await self.db.get_user_by_email(user_data.email)
            if existing_user:
                raise HTTPException(status_code=400, detail="Email already registered")
            
            user_dict = user_data.dict()
            if user_dict.get("password"):
                user_dict["password"] = self.auth.get_password_hash(user_dict["password"])
            
            user = await self.db.create_user(user_dict)
            access_token = self.auth.create_access_token(data={"sub": user.email})
            
            return AuthResponse(
                access_token=access_token,
                token_type="bearer",
                user=user
            )
        
        @self.app.post("/auth/login", response_model=AuthResponse)
        async def login(email: str, password: str):
            user = await self.db.get_user_by_email(email)
            if not user or not self.auth.verify_password(password, getattr(user, 'password', '')):
                raise HTTPException(status_code=400, detail="Invalid credentials")
            
            access_token = self.auth.create_access_token(data={"sub": user.email})
            return AuthResponse(
                access_token=access_token,
                token_type="bearer",
                user=user
            )
        
        # Chat endpoints
        @self.app.get("/chats", response_model=List[Chat])
        async def get_user_chats(current_user: User = Depends(self.get_current_user)):
            return await self.db.get_user_chats(current_user.id)
        
        @self.app.post("/chats", response_model=Chat)
        async def create_chat(chat_data: ChatCreate, current_user: User = Depends(self.get_current_user)):
            chat_dict = chat_data.dict()
            chat_dict["user_id"] = current_user.id
            chat_dict["messages"] = []
            return await self.db.create_chat(chat_dict)
        
        @self.app.post("/chats/{chat_id}/messages")
        async def send_message(
            chat_id: str, 
            message_data: MessageCreate, 
            current_user: User = Depends(self.get_current_user)
        ):
            # In a real implementation, you'd verify the chat belongs to the user
            chat = await self.get_chat(chat_id)
            
            # Add user message
            user_message = Message(
                role="user",
                content=message_data.content,
                images=message_data.images or [],
                timestamp=datetime.utcnow()
            )
            
            # Get AI response based on tool
            if message_data.tool == ToolType.SEARCH:
                ai_response = await AIService.web_search(message_data.content)
            elif message_data.tool == ToolType.STUDY:
                ai_response = await AIService.study_helper(message_data.content)
            elif message_data.tool == ToolType.HOMEWORK:
                ai_response = await AIService.homework_solver(message_data.content)
            elif message_data.tool == ToolType.PROMPT:
                ai_response = await AIService.prompt_generator(message_data.content)
            else:
                messages_history = [{"role": msg.role, "content": msg.content} for msg in chat.messages[-10:]]
                messages_history.append({"role": "user", "content": message_data.content})
                
                ai_response = await AIService.chat_completion(messages_history, message_data.tool)
            
            assistant_message = Message(
                role="assistant",
                content=ai_response["content"],
                sources=ai_response.get("sources", []),
                timestamp=datetime.utcnow()
            )
            
            # Update chat with both messages
            chat.messages.extend([user_message, assistant_message])
            await self.db.update_chat(chat_id, {"messages": chat.messages})
            
            return {
                "user_message": user_message,
                "assistant_message": assistant_message
            }
        
        # Image generation endpoint
        @self.app.post("/images/generate")
        async def generate_image(
            options: ImageGenerationOptions,
            current_user: User = Depends(self.get_current_user)
        ):
            if current_user.credits < 5:
                raise HTTPException(status_code=400, detail="Insufficient credits")
            
            result = await AIService.generate_image(options)
            return result
        
        # User profile endpoint
        @self.app.get("/users/me", response_model=User)
        async def get_current_user_profile(current_user: User = Depends(self.get_current_user)):
            return current_user
    
    async def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(AuthService.security)):
        token_data = self.auth.verify_token(credentials.credentials)
        if not token_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
            )
        user = await self.db.get_user_by_email(token_data.email)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    
    async def get_chat(self, chat_id: str):
        # Mock implementation - in real app, fetch from database
        return Chat(
            id=chat_id,
            user_id="user123",
            title="Test Chat",
            messages=[],
            tool=ToolType.CHAT,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
    
    def run(self, host="0.0.0.0", port=8000):
        uvicorn.run(self.app, host=host, port=port)

# Streamlit Frontend Application
class NovaGPTFrontend:
    def __init__(self, backend_url="http://localhost:8000"):
        self.backend_url = backend_url
        self.setup_page_config()
    
    def setup_page_config(self):
        st.set_page_config(
            page_title="NovaGPT - Your All-in-One AI Universe",
            page_icon="✨",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def local_css(self):
        st.markdown("""
        <style>
        .gradient-bg {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .chat-user {
            background-color: #3B82F6;
            color: white;
            padding: 12px;
            border-radius: 18px 18px 4px 18px;
            margin: 8px 0;
            max-width: 80%;
            margin-left: auto;
        }
        .chat-assistant {
            background-color: #F3F4F6;
            color: #1F2937;
            padding: 12px;
            border-radius: 18px 18px 18px 4px;
            margin: 8px 0;
            max-width: 80%;
            margin-right: auto;
            border: 1px solid #E5E7EB;
        }
        .typing-animation::after {
            content: '|';
            animation: blink 1s infinite;
        }
        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0; }
        }
        </style>
        """, unsafe_allow_html=True)
    
    def init_session_state(self):
        if 'token' not in st.session_state:
            st.session_state.token = None
        if 'user' not in st.session_state:
            st.session_state.user = None
        if 'current_chat' not in st.session_state:
            st.session_state.current_chat = None
        if 'chats' not in st.session_state:
            st.session_state.chats = []
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'selected_tool' not in st.session_state:
            st.session_state.selected_tool = "chat"
    
    def login(self, email: str, password: str) -> bool:
        try:
            response = requests.post(f"{self.backend_url}/auth/login", data={
                "email": email,
                "password": password
            })
            if response.status_code == 200:
                data = response.json()
                st.session_state.token = data["access_token"]
                st.session_state.user = data["user"]
                return True
        except Exception as e:
            st.error(f"Login failed: {str(e)}")
        return False
    
    def register(self, user_data: dict) -> bool:
        try:
            response = requests.post(f"{self.backend_url}/auth/register", json=user_data)
            if response.status_code == 200:
                data = response.json()
                st.session_state.token = data["access_token"]
                st.session_state.user = data["user"]
                return True
            else:
                st.error(response.json().get("detail", "Registration failed"))
        except Exception as e:
            st.error(f"Registration failed: {str(e)}")
        return False
    
    def get_headers(self):
        if st.session_state.token:
            return {"Authorization": f"Bearer {st.session_state.token}"}
        return {}
    
    def load_chats(self):
        try:
            response = requests.get(f"{self.backend_url}/chats", headers=self.get_headers())
            if response.status_code == 200:
                st.session_state.chats = response.json()
        except Exception as e:
            st.error(f"Failed to load chats: {str(e)}")
    
    def create_chat(self, title: str, tool: str = "chat"):
        try:
            response = requests.post(
                f"{self.backend_url}/chats", 
                json={"title": title, "tool": tool},
                headers=self.get_headers()
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"Failed to create chat: {str(e)}")
        return None
    
    def send_message(self, chat_id: str, content: str, tool: str):
        try:
            response = requests.post(
                f"{self.backend_url}/chats/{chat_id}/messages",
                json={"content": content, "tool": tool},
                headers=self.get_headers()
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"Failed to send message: {str(e)}")
        return None
    
    def generate_image(self, prompt: str, style: str = "realistic"):
        try:
            response = requests.post(
                f"{self.backend_url}/images/generate",
                json={
                    "prompt": prompt,
                    "style": style,
                    "size": "1024x1024",
                    "quality": "standard"
                },
                headers=self.get_headers()
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"Failed to generate image: {str(e)}")
        return None
    
    def render_login_page(self):
        st.markdown("""
        <div class="gradient-bg" style="padding: 2rem; border-radius: 1rem; margin-bottom: 2rem;">
            <h1 style="color: white; text-align: center; font-size: 3rem;">✨ NovaGPT</h1>
            <p style="color: white; text-align: center; font-size: 1.2rem;">
            Your All-in-One AI Universe
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login")
                
                if submit:
                    if self.login(email, password):
                        st.success("Login successful!")
                        st.rerun()
        
        with tab2:
            with st.form("register_form"):
                name = st.text_input("Full Name")
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                submit = st.form_submit_button("Register")
                
                if submit:
                    if password != confirm_password:
                        st.error("Passwords don't match")
                    else:
                        user_data = {
                            "name": name,
                            "email": email,
                            "password": password
                        }
                        if self.register(user_data):
                            st.success("Registration successful!")
                            st.rerun()
    
    def render_sidebar(self):
        with st.sidebar:
            st.markdown("""
            <div style="text-align: center; padding: 1rem;">
                <h2 style="color: white;">✨ NovaGPT</h2>
                <p style="color: white; opacity: 0.8;">Your All-in-One AI Universe</p>
            </div>
            """, unsafe_allow_html=True)
            
            # User info
            if st.session_state.user:
                user = st.session_state.user
                st.write(f"Welcome, **{user['name']}**")
                st.write(f"Credits: **{user['credits']}**")
                st.progress(user['credits'] / 100)
            
            # New Chat Button
            if st.button("➕ New Chat", use_container_width=True):
                st.session_state.current_chat = None
                st.session_state.messages = []
                st.rerun()
            
            st.divider()
            
            # Tool Selection
            st.subheader("Tools")
            tool_options = {
                "💬 Chat": "chat",
                "🌐 Web Search": "search", 
                "🎨 Image Creator": "image",
                "📚 Study Helper": "study",
                "📝 Homework Solver": "homework",
                "💡 Prompt Generator": "prompt"
            }
            
            selected_tool = option_menu(
                menu_title=None,
                options=list(tool_options.keys()),
                icons=["chat", "search", "image", "book", "help-circle", "zap"],
                default_index=0,
                styles={
                    "container": {"padding": "0!important", "background-color": "#1f1f1f"},
                    "icon": {"color": "orange", "font-size": "18px"}, 
                    "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#2f2f2f"},
                    "nav-link-selected": {"background-color": "#3f3f3f"},
                }
            )
            
            st.session_state.selected_tool = tool_options[selected_tool]
            
            st.divider()
            
            # Chat History
            st.subheader("Chat History")
            self.load_chats()
            
            for chat in st.session_state.chats[:10]:
                if st.button(f"💬 {chat['title'][:30]}...", key=chat['id'], use_container_width=True):
                    st.session_state.current_chat = chat
                    st.session_state.messages = chat['messages']
                    st.rerun()
            
            st.divider()
            
            # Logout
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.token = None
                st.session_state.user = None
                st.session_state.current_chat = None
                st.session_state.messages = []
                st.rerun()
    
    def render_chat_interface(self):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.header("NovaGPT Assistant")
            st.caption(f"Mode: {st.session_state.selected_tool.title()}")
            
            # Chat messages
            chat_container = st.container()
            
            with chat_container:
                for message in st.session_state.messages:
                    if message['role'] == 'user':
                        st.markdown(f"""
                        <div class="chat-user">
                            {message['content']}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        self.render_assistant_message(message)
            
            # Input area
            input_container = st.container()
            
            with input_container:
                if st.session_state.selected_tool == "image":
                    self.render_image_creator()
                else:
                    self.render_text_input()
        
        with col2:
            self.render_tool_settings()
    
    def render_assistant_message(self, message):
        content = message['content']
        
        # Convert markdown to HTML
        html_content = markdown.markdown(content)
        
        st.markdown(f"""
        <div class="chat-assistant">
            {html_content}
        </div>
        """, unsafe_allow_html=True)
        
        # Show sources if available
        if message.get('sources'):
            with st.expander("📚 Sources"):
                for source in message['sources']:
                    st.write(f"- [{source['title']}]({source['url']})")
    
    def render_text_input(self):
        prompt = st.text_area(
            "Your message:",
            placeholder=f"Ask me anything in {st.session_state.selected_tool} mode...",
            height=100,
            key="text_input"
        )
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("🚀 Send Message", use_container_width=True):
                if prompt.strip():
                    self.handle_text_message(prompt)
        
        with col2:
            if st.button("🔄 Clear", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
    
    def render_image_creator(self):
        st.subheader("🎨 Image Creator")
        
        prompt = st.text_area(
            "Describe your image:",
            placeholder="A beautiful sunset over mountains, digital art...",
            height=100
        )
        
        style = st.selectbox(
            "Style:",
            ["realistic", "anime", "digital-art", "logo", "wallpaper"]
        )
        
        if st.button("✨ Generate Image", use_container_width=True):
            if prompt.strip():
                result = self.generate_image(prompt, style)
                if result and result.get('success'):
                    st.image(result['image_url'], caption=prompt)
                    st.success("Image generated successfully!")
                else:
                    st.error("Failed to generate image")
    
    def render_tool_settings(self):
        st.subheader("Tool Settings")
        
        if st.session_state.selected_tool == "study":
            subject = st.selectbox(
                "Subject:",
                ["General", "Math", "Science", "History", "Literature", "Programming"]
            )
            st.session_state.study_subject = subject
        
        elif st.session_state.selected_tool == "homework":
            show_steps = st.checkbox("Show step-by-step solution", value=True)
            st.session_state.show_steps = show_steps
        
        elif st.session_state.selected_tool == "prompt":
            category = st.selectbox(
                "Prompt Category:",
                ["general", "writing", "coding", "art", "study", "productivity"]
            )
            st.session_state.prompt_category = category
    
    def handle_text_message(self, content):
        # Create new chat if none exists
        if not st.session_state.current_chat:
            chat_title = content[:50] + "..." if len(content) > 50 else content
            new_chat = self.create_chat(chat_title, st.session_state.selected_tool)
            if new_chat:
                st.session_state.current_chat = new_chat
                st.session_state.messages = []
        
        if st.session_state.current_chat:
            # Add user message to UI immediately
            user_message = {
                "role": "user",
                "content": content,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.messages.append(user_message)
            
            # Get AI response
            result = self.send_message(
                st.session_state.current_chat['id'],
                content,
                st.session_state.selected_tool
            )
            
            if result:
                assistant_message = result['assistant_message']
                st.session_state.messages.append(assistant_message)
                st.rerun()
    
    def run(self):
        self.local_css()
        self.init_session_state()
        
        if not st.session_state.token:
            self.render_login_page()
        else:
            self.render_sidebar()
            self.render_chat_interface()

# Main Application Runner
class NovaGPTApp:
    def __init__(self):
        self.backend = None
        self.frontend = None
    
    def run_backend(self, host="0.0.0.0", port=8000):
        """Run the FastAPI backend"""
        print("🚀 Starting NovaGPT Backend...")
        self.backend = NovaGPTBackend()
        self.backend.run(host=host, port=port)
    
    def run_frontend(self):
        """Run the Streamlit frontend"""
        print("🎨 Starting NovaGPT Frontend...")
        self.frontend = NovaGPTFrontend()
        self.frontend.run()
    
    def run_both(self):
        """Run both backend and frontend (requires separate processes)"""
        import multiprocessing
        import time
        
        def run_backend_process():
            self.run_backend()
        
        def run_frontend_process():
            time.sleep(3)  # Wait for backend to start
            self.run_frontend()
        
        # Start backend in a separate process
        backend_process = multiprocessing.Process(target=run_backend_process)
        backend_process.start()
        
        # Start frontend in main process
        run_frontend_process()
        
        backend_process.join()

# Command Line Interface
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="NovaGPT - Your All-in-One AI Universe")
    parser.add_argument("--mode", choices=["backend", "frontend", "both"], default="both",
                       help="Run mode: backend only, frontend only, or both")
    parser.add_argument("--host", default="0.0.0.0", help="Backend host")
    parser.add_argument("--port", type=int, default=8000, help="Backend port")
    
    args = parser.parse_args()
    
    app = NovaGPTApp()
    
    if args.mode == "backend":
        app.run_backend(args.host, args.port)
    elif args.mode == "frontend":
        app.run_frontend()
    else:  # both
        app.run_both()

if __name__ == "__main__":
    # Check if we're running in Streamlit mode
    try:
        from streamlit import runtime
        if runtime.exists():
            # Running in Streamlit
            frontend = NovaGPTFrontend()
            frontend.run()
        else:
            # Running as standalone
            main()
    except ImportError:
        # Running as standalone
        main()
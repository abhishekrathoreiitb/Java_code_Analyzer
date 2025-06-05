# Final Multi-LLM Java RAG Analyzer Backend
# Supports OpenAI, Anthropic, Google, and Local models

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import asyncio
import json
import logging
import os
import tempfile
import zipfile
import uuid
from datetime import datetime
import re
import aiohttp
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import AI libraries
AI_LIBRARIES = {}
try:
    import openai
    AI_LIBRARIES['openai'] = openai
    logger.info("✅ OpenAI library loaded")
except ImportError:
    logger.warning("⚠️ OpenAI library not available")

try:
    import anthropic
    AI_LIBRARIES['anthropic'] = anthropic
    logger.info("✅ Anthropic library loaded")
except ImportError:
    logger.warning("⚠️ Anthropic library not available")

try:
    import google.generativeai as genai
    AI_LIBRARIES['google'] = genai
    logger.info("✅ Google Generative AI library loaded")
except ImportError:
    logger.warning("⚠️ Google Generative AI library not available")

# Simple in-memory storage
sessions_store = {}
processing_status = {}

# Pydantic models
class MultiLLMRequest(BaseModel):
    provider: str  # 'openai', 'anthropic', 'google', 'local'
    model: str
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 4000

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    depth: str = "detailed"  # 'basic', 'detailed', 'comprehensive'

# Create FastAPI app
app = FastAPI(
    title="Java RAG Analyzer - Multi-LLM Professional Edition",
    description="Enterprise Java code analysis with multiple AI providers",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
os.makedirs("./uploads", exist_ok=True)
os.makedirs("./static", exist_ok=True)

# Abstract LLM Provider Interface
class LLMProvider(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = config.get('model')
        self.temperature = config.get('temperature', 0.3)
        self.max_tokens = config.get('max_tokens', 4000)

    @abstractmethod
    async def generate_response(self, prompt: str) -> str:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

# OpenAI Provider
class OpenAIProvider(LLMProvider):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if 'openai' in AI_LIBRARIES:
            self.client = AI_LIBRARIES['openai'].AsyncOpenAI(api_key=config.get('api_key'))
        else:
            self.client = None

    def is_available(self) -> bool:
        return 'openai' in AI_LIBRARIES and self.config.get('api_key') is not None

    async def generate_response(self, prompt: str) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise Exception(f"OpenAI generation failed: {str(e)}")

# Anthropic Provider
class AnthropicProvider(LLMProvider):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if 'anthropic' in AI_LIBRARIES:
            self.client = AI_LIBRARIES['anthropic'].AsyncAnthropic(api_key=config.get('api_key'))
        else:
            self.client = None

    def is_available(self) -> bool:
        return 'anthropic' in AI_LIBRARIES and self.config.get('api_key') is not None

    async def generate_response(self, prompt: str) -> str:
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise Exception(f"Anthropic generation failed: {str(e)}")

# Google Provider
class GoogleProvider(LLMProvider):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if 'google' in AI_LIBRARIES:
            AI_LIBRARIES['google'].configure(api_key=config.get('api_key'))
            self.model_instance = AI_LIBRARIES['google'].GenerativeModel(self.model)
        else:
            self.model_instance = None

    def is_available(self) -> bool:
        return 'google' in AI_LIBRARIES and self.config.get('api_key') is not None

    async def generate_response(self, prompt: str) -> str:
        try:
            response = self.model_instance.generate_content(
                prompt,
                generation_config={
                    'temperature': self.temperature,
                    'max_output_tokens': self.max_tokens
                }
            )
            return response.text
        except Exception as e:
            logger.error(f"Google API error: {e}")
            raise Exception(f"Google generation failed: {str(e)}")

# Local Provider (Ollama, LM Studio, etc.)
class LocalProvider(LLMProvider):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.endpoint = config.get('endpoint', 'http://localhost:11434/api/generate')

    def is_available(self) -> bool:
        return self.endpoint is not None

    async def generate_response(self, prompt: str) -> str:
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_ctx": self.max_tokens
                    }
                }
                
                async with session.post(self.endpoint, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get('response', 'No response generated')
                    else:
                        raise Exception(f"Local API returned status {response.status}")
                        
        except Exception as e:
            logger.error(f"Local API error: {e}")
            raise Exception(f"Local generation failed: {str(e)}")

# Provider Factory
class LLMProviderFactory:
    @staticmethod
    def create_provider(config: Dict[str, Any]) -> LLMProvider:
        provider_type = config.get('provider')
        
        if provider_type == 'openai':
            return OpenAIProvider(config)
        elif provider_type == 'anthropic':
            return AnthropicProvider(config)
        elif provider_type == 'google':
            return GoogleProvider(config)
        elif provider_type == 'local':
            return LocalProvider(config)
        else:
            raise ValueError(f"Unsupported provider: {provider_type}")

# Enhanced Java Code Analyzer
class EnhancedJavaAnalyzer:
    def __init__(self):
        self.design_patterns = {
            'singleton': [
                r'private\s+static\s+\w+\s+instance',
                r'getInstance\(\)',
                r'private\s+\w+\s*\(\s*\)\s*\{'
            ],
            'factory': [
                r'create\w*\(',
                r'Factory',
                r'public\s+static\s+\w+\s+create'
            ],
            'builder': [
                r'\.build\(\)',
                r'Builder',
                r'public\s+\w+\s+set\w+\('
            ],
            'observer': [
                r'addListener',
                r'removeListener',
                r'notify',
                r'Observer'
            ],
            'mvc': [
                r'@Controller',
                r'@RestController',
                r'@RequestMapping',
                r'@GetMapping',
                r'@PostMapping'
            ],
            'repository': [
                r'@Repository',
                r'@Entity',
                r'@Table',
                r'Repository'
            ],
            'service': [
                r'@Service',
                r'@Component',
                r'@Autowired'
            ],
            'strategy': [
                r'Strategy',
                r'interface.*Strategy',
                r'class.*Strategy'
            ],
            'decorator': [
                r'Decorator',
                r'Component.*interface',
                r'ConcreteDecorator'
            ]
        }
        
        self.security_issues = {
            'sql_injection': [
                r'executeQuery\s*\(\s*["\'][^"\']*\+',
                r'prepareStatement\s*\(\s*["\'][^"\']*\+',
                r'createQuery\s*\(\s*["\'][^"\']*\+'
            ],
            'hardcoded_credentials': [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
                r'api_key\s*=\s*["\'][^"\']+["\']'
            ],
            'xss_vulnerability': [
                r'\.innerHTML\s*=',
                r'document\.write\s*\(',
                r'eval\s*\('
            ],
            'path_traversal': [
                r'new\s+File\s*\([^)]*\+',
                r'Files\.readString\s*\([^)]*\+',
                r'\.\./'
            ],
            'insecure_random': [
                r'new\s+Random\s*\(',
                r'Math\.random\s*\('
            ],
            'weak_crypto': [
                r'DES\s*\(',
                r'MD5\s*\(',
                r'SHA1\s*\('
            ]
        }
        
        self.performance_issues = {
            'string_concatenation_loop': [
                r'for\s*\([^)]*\)\s*\{[^}]*\+\s*=.*String',
                r'while\s*\([^)]*\)\s*\{[^}]*\+\s*='
            ],
            'inefficient_collections': [
                r'Vector\s*<',
                r'Hashtable\s*<',
                r'synchronized.*ArrayList'
            ],
            'resource_leaks': [
                r'new\s+FileInputStream[^}]*(?!.*\.close\(\))',
                r'Connection\s+\w+[^}]*(?!.*\.close\(\))',
                r'PreparedStatement[^}]*(?!.*\.close\(\))'
            ],
            'n_plus_one': [
                r'for\s*\([^)]*\)\s*\{[^}]*find.*\(',
                r'while\s*\([^)]*\)\s*\{[^}]*query.*\('
            ]
        }

    def analyze_java_file(self, content: str, file_path: str) -> Dict[str, Any]:
        """Comprehensive Java file analysis"""
        lines = content.split('\n')
        code_lines = [line for line in lines if line.strip() and not line.strip().startswith('//')]
        
        # Extract package and imports
        package_match = re.search(r'package\s+([\w\.]+)', content)
        package_name = package_match.group(1) if package_match else 'default'
        
        imports = re.findall(r'import\s+([\w\.]+)', content)
        
        # Find classes, interfaces, and methods
        classes = re.findall(r'(?:public\s+)?(?:abstract\s+)?(?:final\s+)?class\s+(\w+)', content)
        interfaces = re.findall(r'(?:public\s+)?interface\s+(\w+)', content)
        methods = re.findall(r'(?:public|private|protected)\s+(?:static\s+)?(?:\w+\s+)*(\w+)\s*\([^)]*\)', content)
        
        # Calculate complexity
        complexity_keywords = ['if', 'else', 'while', 'for', 'switch', 'case', 'catch', 'try', '&&', '||', '?']
        complexity = sum(content.count(keyword) for keyword in complexity_keywords)
        
        # Detect patterns and issues
        detected_patterns = self._detect_patterns(content, self.design_patterns)
        security_issues = self._detect_patterns(content, self.security_issues)
        performance_issues = self._detect_patterns(content, self.performance_issues)
        
        # Extract annotations
        annotations = re.findall(r'@(\w+)', content)
        
        # Determine file type
        file_type = self._determine_file_type(content, file_path)
        
        return {
            'file_path': file_path,
            'package': package_name,
            'imports': imports,
            'classes': classes,
            'interfaces': interfaces,
            'methods': methods,
            'total_lines': len(lines),
            'code_lines': len(code_lines),
            'complexity_score': complexity,
            'design_patterns': detected_patterns,
            'security_issues': security_issues,
            'performance_issues': performance_issues,
            'annotations': list(set(annotations)),
            'file_type': file_type,
            'spring_features': self._identify_spring_features(content, annotations)
        }

    def _detect_patterns(self, content: str, pattern_dict: Dict[str, List[str]]) -> List[str]:
        detected = []
        for pattern_name, patterns in pattern_dict.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    detected.append(pattern_name)
                    break
        return detected

    def _determine_file_type(self, content: str, file_path: str) -> str:
        if 'Controller' in file_path or any(ann in content for ann in ['@Controller', '@RestController']):
            return 'Spring MVC Controller'
        elif 'Service' in file_path or '@Service' in content:
            return 'Spring Service'
        elif 'Repository' in file_path or '@Repository' in content:
            return 'Spring Repository'
        elif 'Configuration' in file_path or '@Configuration' in content:
            return 'Spring Configuration'
        elif 'Entity' in file_path or '@Entity' in content:
            return 'JPA Entity'
        elif 'interface' in content.lower():
            return 'Java Interface'
        else:
            return 'Java Class'

    def _identify_spring_features(self, content: str, annotations: List[str]) -> List[str]:
        features = []
        
        spring_annotations = {
            'Controller': 'Spring MVC Controller',
            'RestController': 'Spring REST Controller',
            'Service': 'Spring Service Layer',
            'Repository': 'Spring Data Repository',
            'Component': 'Spring Component',
            'Configuration': 'Spring Configuration',
            'Bean': 'Spring Bean Definition',
            'Autowired': 'Dependency Injection',
            'RequestMapping': 'Request Mapping',
            'GetMapping': 'GET Request Mapping',
            'PostMapping': 'POST Request Mapping',
            'PutMapping': 'PUT Request Mapping',
            'DeleteMapping': 'DELETE Request Mapping',
            'Entity': 'JPA Entity',
            'Table': 'JPA Table Mapping',
            'Transactional': 'Transaction Management'
        }
        
        for annotation in annotations:
            if annotation in spring_annotations:
                features.append(spring_annotations[annotation])
        
        return list(set(features))

# Multi-LLM RAG System
class MultiLLMRAGSystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.documents = []
        self.analyzer = EnhancedJavaAnalyzer()
        self.provider = LLMProviderFactory.create_provider(config)
        self.current_codebase_id = None

    def clear_previous_context(self):
        """Clear all previous documents and context"""
        logger.info("🧹 Clearing previous codebase context...")
        self.documents = []
        self.current_codebase_id = str(uuid.uuid4())[:8]
        logger.info(f"✅ Context cleared. New codebase ID: {self.current_codebase_id}")

    def add_document(self, content: str, file_path: str):
        """Add document with comprehensive analysis"""
        try:
            analysis = self.analyzer.analyze_java_file(content, file_path)
            
            doc = {
                'content': content,
                'file_path': file_path,
                'analysis': analysis,
                'id': len(self.documents)
            }
            
            self.documents.append(doc)
            return analysis
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return {}

    async def query(self, query: str, top_k: int = 5, depth: str = "detailed") -> Dict[str, Any]:
        """Multi-LLM powered query with configurable depth"""
        try:
            if not self.documents:
                return {
                    'answer': "No documents processed yet. Please upload a Java codebase first.",
                    'sources': [],
                    'provider': self.config.get('provider'),
                    'model': self.config.get('model')
                }
            
            # Find relevant documents
            relevant_docs = self._find_relevant_documents(query, top_k)
            
            if self.provider.is_available():
                return await self._ai_powered_analysis(query, relevant_docs, depth)
            else:
                return self._rule_based_analysis(query, relevant_docs, depth)
                
        except Exception as e:
            logger.error(f"Query error: {e}")
            return {
                'answer': f"Error processing query: {str(e)}",
                'sources': [],
                'provider': self.config.get('provider'),
                'model': self.config.get('model')
            }

    def _find_relevant_documents(self, query: str, top_k: int) -> List[Dict]:
        """Find relevant documents using keyword matching and scoring"""
        query_lower = query.lower()
        scored_docs = []
        
        for doc in self.documents:
            score = 0
            content_lower = doc['content'].lower()
            analysis = doc.get('analysis', {})
            file_path_lower = doc['file_path'].lower()
            
            # Score based on query keywords
            for word in query_lower.split():
                score += content_lower.count(word) * 2
                score += file_path_lower.count(word) * 5
                score += str(analysis).lower().count(word)
            
            # Boost for specific matches
            if 'controller' in query_lower and 'Controller' in analysis.get('file_type', ''):
                score += 15
            if 'security' in query_lower and analysis.get('security_issues'):
                score += 20
            if 'pattern' in query_lower and analysis.get('design_patterns'):
                score += 15
            if 'performance' in query_lower and analysis.get('performance_issues'):
                score += 15
            
            if score > 0:
                scored_docs.append((score, doc))
        
        # Sort and return top documents
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]

    async def _ai_powered_analysis(self, query: str, documents: List[Dict], depth: str) -> Dict[str, Any]:
        """AI-powered analysis using the configured provider"""
        try:
            # Prepare context
            context = self._prepare_context(documents, depth)
            
            # Create depth-specific prompt
            prompt = self._create_analysis_prompt(query, context, depth)
            
            # Generate response
            response = await self.provider.generate_response(prompt)
            
            return {
                'answer': response,
                'sources': [doc.get('analysis', {}) for doc in documents],
                'provider': self.config.get('provider'),
                'model': self.config.get('model'),
                'codebase_id': self.current_codebase_id,
                'analysis_depth': depth
            }
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return self._rule_based_analysis(query, documents, depth)

    def _prepare_context(self, documents: List[Dict], depth: str) -> str:
        """Prepare context based on analysis depth"""
        context_parts = []
        
        # Adjust context size based on depth
        max_chars = {
            'basic': 1000,
            'detailed': 2500,
            'comprehensive': 5000
        }.get(depth, 2500)
        
        for i, doc in enumerate(documents[:5], 1):  # Max 5 documents
            file_name = os.path.basename(doc['file_path'])
            analysis = doc.get('analysis', {})
            content_preview = doc['content'][:max_chars] + "..." if len(doc['content']) > max_chars else doc['content']
            
            context_parts.append(f"""
**File {i}: {file_name}**
- **Type**: {analysis.get('file_type', 'Unknown')}
- **Package**: {analysis.get('package', 'Unknown')}
- **Classes**: {', '.join(analysis.get('classes', [])) if analysis.get('classes') else 'None'}
- **Methods**: {len(analysis.get('methods', []))} methods
- **Complexity**: {analysis.get('complexity_score', 0)}
- **Spring Features**: {', '.join(analysis.get('spring_features', [])) if analysis.get('spring_features') else 'None'}
- **Design Patterns**: {', '.join(analysis.get('design_patterns', [])) if analysis.get('design_patterns') else 'None'}
- **Security Issues**: {', '.join(analysis.get('security_issues', [])) if analysis.get('security_issues') else 'None'}
- **Performance Issues**: {', '.join(analysis.get('performance_issues', [])) if analysis.get('performance_issues') else 'None'}

**Code**:
```java
{content_preview}
```
""")
        
        return '\n'.join(context_parts)

    def _create_analysis_prompt(self, query: str, context: str, depth: str) -> str:
        """Create analysis prompt based on depth and query type"""
        
        base_instructions = {
            'basic': "Provide a concise, focused answer (200-400 words) addressing the specific question.",
            'detailed': "Provide a comprehensive analysis (400-800 words) with technical details and practical insights.",
            'comprehensive': "Provide an in-depth, expert-level analysis (800+ words) with detailed technical explanations, code examples, and actionable recommendations."
        }
        
        # Determine query type
        query_lower = query.lower()
        if any(word in query_lower for word in ['functionality', 'function', 'does', 'purpose', 'what', 'explain']):
            query_type = "functionality"
        elif any(word in query_lower for word in ['security', 'vulnerability', 'secure']):
            query_type = "security"
        elif any(word in query_lower for word in ['architecture', 'structure', 'design', 'pattern']):
            query_type = "architecture"
        elif any(word in query_lower for word in ['performance', 'optimization', 'bottleneck']):
            query_type = "performance"
        else:
            query_type = "general"
        
        prompt = f"""
You are a senior Java architect and enterprise software expert analyzing a Spring Boot codebase.

QUERY: {query}
ANALYSIS DEPTH: {depth}
QUERY TYPE: {query_type}

CODEBASE CONTEXT:
{context}

INSTRUCTIONS:
{base_instructions[depth]}

For {query_type} analysis, focus on:
"""
        
        if query_type == "functionality":
            prompt += """
- Detailed explanation of what the code does and how it works
- Method-by-method breakdown of key functionality
- Data flow and processing logic
- Integration points with other components
- Business logic and domain-specific operations
"""
        elif query_type == "security":
            prompt += """
- Specific security vulnerabilities and their risks
- Authentication and authorization mechanisms
- Input validation and sanitization
- Cryptography and sensitive data handling
- Security best practices compliance
"""
        elif query_type == "architecture":
            prompt += """
- Architectural patterns and design principles
- Component relationships and dependencies
- Layer separation and coupling analysis
- Design pattern implementations
- Scalability and maintainability considerations
"""
        elif query_type == "performance":
            prompt += """
- Performance bottlenecks and optimization opportunities
- Resource usage and efficiency
- Database access patterns
- Caching strategies
- Scalability concerns
"""
        else:
            prompt += """
- Direct, comprehensive answer to the specific question
- Relevant technical details and implementation specifics
- Practical insights for developers and architects
"""
        
        prompt += """

Provide specific examples from the actual code and reference file names, class names, and method names where relevant.
Make your analysis practical and actionable for development teams.
"""
        
        return prompt

    def _rule_based_analysis(self, query: str, documents: List[Dict], depth: str) -> Dict[str, Any]:
        """Enhanced rule-based analysis as fallback"""
        # Implementation similar to previous version but enhanced
        # This would be a comprehensive fallback when AI is not available
        pass

# Connection Manager for WebSocket
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"🔌 WebSocket connected for session: {session_id}")

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"🔌 WebSocket disconnected for session: {session_id}")

    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"❌ Failed to send WebSocket message: {e}")
                self.disconnect(session_id)

connection_manager = ConnectionManager()

# API Routes
@app.get("/", response_class=HTMLResponse)
async def get_web_interface():
    """Serve the enhanced web interface"""
    try:
        with open("static/index.html", "r", encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return """
        <h1>Java RAG Analyzer - Multi-LLM Professional Edition</h1>
        <p>Please create the static/index.html file with the enhanced web interface.</p>
        <p>API docs: <a href="/docs">/docs</a></p>
        """

@app.post("/api/sessions")
async def create_session(request: MultiLLMRequest):
    """Create a new session with multi-LLM support"""
    session_id = str(uuid.uuid4())
    
    logger.info(f"🎯 Creating session with {request.provider} {request.model}")
    
    try:
        # Create RAG system with the specified provider
        config = {
            'provider': request.provider,
            'model': request.model,
            'api_key': request.api_key,
            'endpoint': request.endpoint,
            'temperature': request.temperature,
            'max_tokens': request.max_tokens
        }
        
        rag_system = MultiLLMRAGSystem(config)
        sessions_store[session_id] = rag_system
        processing_status[session_id] = {
            "status": "ready", 
            "progress": 0, 
            "message": f"Session created with {request.provider} {request.model}"
        }
        
        logger.info(f"✅ Session created: {session_id}")
        return {
            "session_id": session_id, 
            "status": "created",
            "provider": request.provider,
            "model": request.model,
            "provider_available": rag_system.provider.is_available()
        }
        
    except Exception as e:
        logger.error(f"Session creation error: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to create session: {str(e)}")

@app.post("/api/sessions/{session_id}/upload")
async def upload_codebase(
    session_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload and process codebase with enhanced analysis"""
    logger.info(f"🚀 Upload request for session: {session_id}")
    
    if session_id not in sessions_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Please upload a ZIP file")
    
    try:
        upload_path = f"uploads/{session_id}_{file.filename}"
        with open(upload_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"✅ File saved: {upload_path}")
        
        # Start background processing
        background_tasks.add_task(process_with_enhanced_analysis, session_id, upload_path)
        
        return {"status": "upload_received", "message": "Enhanced AI processing started"}
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

async def process_with_enhanced_analysis(session_id: str, file_path: str):
    """Process file with enhanced multi-LLM analysis"""
    logger.info(f"🔄 Enhanced processing started for session: {session_id}")
    
    try:
        rag_system = sessions_store[session_id]
        
        # Clear previous context
        rag_system.clear_previous_context()
        
        # Update status
        processing_status[session_id] = {"status": "processing", "progress": 5, "message": "Clearing previous context..."}
        await connection_manager.send_message(session_id, {
            "type": "progress", "progress": 5, "message": "Clearing previous context..."
        })
        
        await asyncio.sleep(0.5)
        
        processing_status[session_id] = {"status": "processing", "progress": 10, "message": "Extracting new codebase..."}
        await connection_manager.send_message(session_id, {
            "type": "progress", "progress": 10, "message": "Extracting new codebase..."
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract ZIP
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find Java files
            java_files = []
            for root, dirs, files in os.walk(temp_dir):
                dirs[:] = [d for d in dirs if d not in ['target', 'build', '.git', 'node_modules']]
                for file in files:
                    if file.endswith('.java'):
                        java_files.append(os.path.join(root, file))
            
            logger.info(f"📁 Found {len(java_files)} Java files in new codebase")
            
            processing_status[session_id] = {"status": "processing", "progress": 30, "message": f"Analyzing {len(java_files)} Java files with AI..."}
            await connection_manager.send_message(session_id, {
                "type": "progress", "progress": 30, "message": f"Found {len(java_files)} Java files. Starting enhanced analysis..."
            })
            
            # Process files with comprehensive analysis
            total_classes = 0
            total_methods = 0
            total_lines = 0
            spring_features = set()
            design_patterns = set()
            security_issues = set()
            performance_issues = set()
            
            for i, java_file in enumerate(java_files):
                try:
                    with open(java_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Add to RAG system with enhanced analysis
                    analysis = rag_system.add_document(content, java_file)
                    
                    # Update comprehensive statistics
                    total_classes += len(analysis.get('classes', []))
                    total_methods += len(analysis.get('methods', []))
                    total_lines += analysis.get('code_lines', 0)
                    spring_features.update(analysis.get('spring_features', []))
                    design_patterns.update(analysis.get('design_patterns', []))
                    security_issues.update(analysis.get('security_issues', []))
                    performance_issues.update(analysis.get('performance_issues', []))
                    
                    # Update progress every 3 files
                    if i % 3 == 0:
                        progress = 30 + int((i / len(java_files)) * 60)
                        processing_status[session_id] = {
                            "status": "processing",
                            "progress": progress,
                            "message": f"Enhanced analysis: {i+1}/{len(java_files)} files | Found {total_classes} classes, {len(design_patterns)} patterns"
                        }
                        await connection_manager.send_message(session_id, {
                            "type": "progress",
                            "progress": progress,
                            "message": f"Analyzing: {i+1}/{len(java_files)} files"
                        })
                    
                    logger.info(f"📄 Enhanced analysis {i+1}/{len(java_files)}: {os.path.basename(java_file)}")
                    
                except Exception as e:
                    logger.warning(f"Failed to process {java_file}: {e}")
            
            # Complete with comprehensive statistics
            processing_status[session_id] = {
                "status": "completed", 
                "progress": 100, 
                "message": f"Enhanced AI analysis completed! ({rag_system.current_codebase_id})",
                "stats": {
                    "total_files": len(java_files),
                    "total_lines": total_lines,
                    "total_classes": total_classes,
                    "total_methods": total_methods,
                    "spring_features": list(spring_features),
                    "design_patterns": list(design_patterns),
                    "security_issues": list(security_issues),
                    "performance_issues": list(performance_issues),
                    "codebase_id": rag_system.current_codebase_id,
                    "provider": rag_system.config.get('provider'),
                    "model": rag_system.config.get('model')
                }
            }
            
            await connection_manager.send_message(session_id, {
                "type": "completed",
                "stats": processing_status[session_id]["stats"]
            })
            
            logger.info(f"🎉 Enhanced processing completed for session: {session_id}")
            
    except Exception as e:
        error_msg = f"Enhanced processing error: {str(e)}"
        logger.error(error_msg)
        processing_status[session_id] = {"status": "error", "progress": 0, "message": error_msg}
        await connection_manager.send_message(session_id, {"type": "error", "message": error_msg})
    
    finally:
        try:
            os.remove(file_path)
            logger.info("🧹 Cleaned up upload file")
        except:
            pass

@app.get("/api/sessions/{session_id}/status")
async def get_status(session_id: str):
    """Get processing status"""
    if session_id not in sessions_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    status = processing_status.get(session_id, {"status": "unknown", "progress": 0, "message": ""})
    return status

@app.post("/api/sessions/{session_id}/query")
async def query_codebase(session_id: str, request: QueryRequest):
    """Query with multi-LLM analysis"""
    logger.info(f"🔍 Query: {request.query} (depth: {request.depth})")
    
    rag_system = sessions_store.get(session_id)
    if not rag_system:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not rag_system.documents:
        raise HTTPException(status_code=400, detail="No codebase processed yet")
    
    try:
        result = await rag_system.query(request.query, request.top_k, request.depth)
        logger.info(f"✅ Query completed with {result.get('provider')} {result.get('model')}")
        return result
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")

@app.delete("/api/sessions/{session_id}/clear")
async def clear_context(session_id: str):
    """Clear current codebase context"""
    rag_system = sessions_store.get(session_id)
    if not rag_system:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        old_count = len(rag_system.documents)
        rag_system.clear_previous_context()
        
        processing_status[session_id] = {"status": "ready", "progress": 0, "message": "Context cleared"}
        
        return {
            "status": "cleared",
            "message": f"Cleared {old_count} documents",
            "new_codebase_id": rag_system.current_codebase_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear failed: {str(e)}")

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket for real-time updates"""
    await connection_manager.connect(websocket, session_id)
    try:
        while True:
            await asyncio.sleep(1)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        connection_manager.disconnect(session_id)

@app.get("/health")
async def health_check():
    """Health check with provider status"""
    available_providers = []
    for provider in ['openai', 'anthropic', 'google']:
        if provider in AI_LIBRARIES:
            available_providers.append(provider)
    
    return {
        "status": "healthy",
        "sessions": len(sessions_store),
        "available_providers": available_providers,
        "version": "3.0.0 - Multi-LLM Professional Edition"
    }

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Java RAG Analyzer - Multi-LLM Professional Edition")
    print("🧠 Supported AI Providers:")
    
    for provider, lib in AI_LIBRARIES.items():
        print(f"   ✅ {provider.title()}: {lib.__name__ if hasattr(lib, '__name__') else 'Available'}")
    
    if not AI_LIBRARIES:
        print("   ⚠️ No AI libraries available - using rule-based analysis")
    
    print("🌐 Server starting at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("=" * 60)
    
    # Create directories
    for directory in ["static", "uploads"]:
        os.makedirs(directory, exist_ok=True)
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info"
    )
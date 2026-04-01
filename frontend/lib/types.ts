export interface UserResponse {
  id: number;
  username: string;
  email: string;
  is_active: boolean;
  created_at?: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  user: UserResponse;
}

export interface SourceDocument {
  content: string;
  source: string;
  page?: number;
  score?: number;
}

export interface QueryResponse {
  answer: string;
  sources: SourceDocument[];
  confident: boolean;
  confidence_score: number;
  from_cache: boolean;
  query_rewritten?: string;
  intent?: string;
  response_time_ms: number;
  timestamp: string;
}

export interface DocumentInfo {
  filename: string;
  size_kb: number;
  chunks_created?: number;
  modified?: string;
  uploaded_at?: string;
}

export interface SystemStats {
  total_queries: number;
  cache_hit_rate: number;
  cache_stats: { keys: number; status: string };
  vectorstore_loaded: boolean;
  bm25_loaded?: boolean;
  uptime_seconds: number;
  total_queries_all_time?: number;
}

export interface Message {
  id: string;
  role: "user" | "bot";
  content: string;
  sources?: SourceDocument[];
  confident?: boolean;
  confidence_score?: number;
  intent?: string;
  query_rewritten?: string;
  from_cache?: boolean;
  response_time_ms?: number;
  isStreaming?: boolean;
}

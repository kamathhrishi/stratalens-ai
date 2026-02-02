# StrataLens React/TypeScript Migration Plan

## Executive Summary

This document provides a comprehensive migration plan for converting the StrataLens frontend from vanilla JavaScript/HTML/CSS to a modern React/TypeScript stack with Tailwind CSS. The migration aims to:

1. **Reduce code redundancy** - Consolidate ~10 major redundancy patterns identified
2. **Improve extensibility** - Component-based architecture with proper state management
3. **Enhance design** - Professional UI using Tailwind CSS and shadcn/ui components
4. **Improve developer experience** - Type safety, better tooling, and maintainability

---

## Table of Contents

1. [Current Architecture Analysis](#current-architecture-analysis)
2. [Feature Inventory](#feature-inventory)
3. [Recommended Tech Stack](#recommended-tech-stack)
4. [Project Structure](#project-structure)
5. [Component Architecture](#component-architecture)
6. [State Management Strategy](#state-management-strategy)
7. [API Layer & TypeScript Interfaces](#api-layer--typescript-interfaces)
8. [Redundancy Elimination Plan](#redundancy-elimination-plan)
9. [Design System & Tailwind Configuration](#design-system--tailwind-configuration)
10. [Migration Phases](#migration-phases)
11. [Testing Strategy](#testing-strategy)

---

## Current Architecture Analysis

### File Structure Overview

| File | Lines | Purpose |
|------|-------|---------|
| `index.html` | 2,724 | Main app HTML with 8 sections, 17+ modals |
| `app.js` | 10,000+ | Main application logic, section switching |
| `chat.js` | 4,539 | Chat interface, SSE streaming, markdown |
| `styles.css` | 6,016 | CSS with 40+ custom properties |
| `auth.js` | 541 | Clerk authentication |
| `utils.js` | 887 | Formatting utilities |
| `charts.js` | ~500 | Chart rendering with Chart.js |
| `session.js` | ~200 | Session management |
| `config.js` | ~100 | Configuration |
| `landing.html` | 76 KB | Landing page |
| `landing-integration.js` | 436 | Landing-to-app transition |

### Current Technology Stack

- **JavaScript**: Vanilla JS with jQuery for DOM manipulation
- **HTML**: Single monolithic HTML file with inline templates
- **CSS**: Custom CSS with CSS variables for theming
- **Charts**: Chart.js for visualizations
- **Markdown**: marked.js + DOMPurify for rendering
- **Authentication**: Clerk (already integrated)
- **Data Tables**: Custom HTML table generation

### Identified Pain Points

1. **Monolithic files** - app.js is 10,000+ lines with mixed concerns
2. **Global state** - Heavy reliance on global variables
3. **DOM manipulation** - Direct DOM manipulation everywhere
4. **Code duplication** - 10+ redundancy patterns identified
5. **No type safety** - Runtime errors from type mismatches
6. **Difficult testing** - Tightly coupled code

---

## Feature Inventory

### 1. Chat Section (`chat`)

| Feature | Description | API Endpoints |
|---------|-------------|---------------|
| Chat Messages | Send/receive messages with AI | `POST /chat/stream` |
| Mode Switching | Ask vs Agent modes | Query param `mode` |
| Streaming Responses | SSE-based real-time responses | Server-Sent Events |
| Message History | Load previous conversations | `GET /chat/history` |
| Markdown Rendering | Rich text with code highlighting | Client-side |
| Ticker Autocomplete | Search companies while typing | `GET /companies/search` |
| Message Actions | Copy, retry, feedback | Various |
| Table Rendering | Format JSON as interactive tables | Client-side |
| Chart Rendering | Inline charts in responses | Chart.js |
| Export Chat | Export conversation history | `GET /chat/export` |
| Clear History | Clear conversation | `DELETE /chat/history` |

**Chat UI Components:**
- ChatInput with character counter
- MessageList with virtualization needed
- MessageBubble (user/assistant variants)
- ModeSelector dropdown
- TypingIndicator
- ErrorMessage
- StreamingText

### 2. Companies Section (`companies`)

| Feature | Description | API Endpoints |
|---------|-------------|---------------|
| Company Search | Autocomplete company search | `GET /companies/search` |
| Company Profile | Full company details | `GET /companies/{ticker}/profile` |
| Financial Overview | Key financials display | `GET /companies/{ticker}/overview` |
| Segment Data | Business segment breakdown | `GET /companies/{ticker}/segments` |
| Company Comparison | Compare multiple companies | `GET /companies/compare` |
| Peer Analysis | Industry peer comparison | `GET /companies/{ticker}/peers` |
| Save to Collection | Add company to collection | `POST /screens/companies` |

**Company UI Components:**
- CompanySearch with autocomplete
- CompanyCard
- CompanyProfile
- FinancialOverview
- SegmentChart
- PeerComparison
- CompanyTable

### 3. Screener Section (`screener`)

| Feature | Description | API Endpoints |
|---------|-------------|---------------|
| Natural Language Query | Query stocks with NL | `POST /screener/query/stream` |
| Results Table | Paginated, sortable results | Streamed response |
| Column Sorting | Multi-column sort | Client-side + server |
| Pagination | Navigate large result sets | Query params |
| Save Screen | Save query as screen | `POST /screens` |
| Export Results | Export to CSV/Excel | Client-side |
| Column Visibility | Show/hide columns | Client-side |
| Conditional Formatting | Color-coded values | Client-side |

**Screener UI Components:**
- ScreenerInput
- ResultsTable with TanStack Table
- ColumnSelector
- PaginationControls
- SortIndicator
- SaveScreenModal
- ExportButton

### 4. Collections Section (`collections`)

| Feature | Description | API Endpoints |
|---------|-------------|---------------|
| List Screens | View saved screens | `GET /screens` |
| Load Screen | Load saved screen | `GET /screens/{id}` |
| Delete Screen | Remove saved screen | `DELETE /screens/{id}` |
| Rename Screen | Edit screen name | `PUT /screens/{id}` |
| Duplicate Screen | Clone existing screen | `POST /screens/{id}/duplicate` |
| Share Screen | Share with others | Future feature |

**Collections UI Components:**
- ScreenList
- ScreenCard
- ScreenActions
- CreateScreenModal
- DeleteConfirmModal

### 5. Charting Section (`charting`)

| Feature | Description | API Endpoints |
|---------|-------------|---------------|
| Multi-Company Charts | Compare metrics | `GET /charting/multi-company` |
| Time Period Selection | Date range picker | Query params |
| Metric Selection | Choose financial metrics | Query params |
| Chart Types | Line, bar, area charts | Client-side |
| Normalize Toggle | Normalize for comparison | Client-side |
| Export Chart | Download as image | Client-side |
| Full Screen Mode | Expanded chart view | Client-side |

**Charting UI Components:**
- ChartContainer
- MetricSelector
- CompanyMultiSelect
- DateRangePicker
- ChartTypeSelector
- ChartLegend
- ExportChartButton

### 6. Authentication & User Management

| Feature | Description | Implementation |
|---------|-------------|----------------|
| Sign In | Clerk sign in modal | Clerk SDK |
| Sign Up | Clerk registration | Clerk SDK |
| Sign Out | Log out user | Clerk SDK |
| User Profile | View/edit profile | Clerk SDK |
| Session Management | Token refresh | Clerk SDK |
| Auth Guards | Protected routes | React Router |

**Auth UI Components:**
- AuthProvider (Clerk)
- SignInButton
- UserButton
- ProtectedRoute

### 7. Navigation & Layout

| Feature | Description |
|---------|-------------|
| Sidebar Navigation | Section switching |
| Responsive Layout | Mobile/tablet support |
| Dark Mode | Theme toggle (CSS vars exist) |
| Breadcrumbs | Navigation context |
| Toast Notifications | Success/error messages |
| Loading States | Skeletons and spinners |

**Layout UI Components:**
- AppShell
- Sidebar
- Header
- Footer
- MobileNav
- ThemeToggle
- ToastProvider

### 8. Modals Inventory

| Modal | Purpose | Trigger |
|-------|---------|---------|
| AuthModal | Login/Register | Nav button |
| SaveScreenModal | Save screener query | Save button |
| CompanyDetailModal | Full company view | Company click |
| ExportModal | Export options | Export button |
| ConfirmDeleteModal | Confirm deletion | Delete action |
| SettingsModal | User preferences | Settings icon |
| FeedbackModal | Submit feedback | Feedback button |
| ShareModal | Share content | Share button |
| ChartFullscreenModal | Expanded chart | Expand button |
| KeyboardShortcutsModal | Help | ? key |

---

## Recommended Tech Stack

### Core Framework

| Package | Version | Purpose |
|---------|---------|---------|
| `react` | ^18.3 | UI framework |
| `react-dom` | ^18.3 | DOM rendering |
| `typescript` | ^5.4 | Type safety |
| `vite` | ^5.4 | Build tool & dev server |

### State Management

| Package | Purpose | Use Case |
|---------|---------|----------|
| `zustand` | Global state | UI state, user preferences |
| `@tanstack/react-query` | Server state | API data, caching |
| `immer` | Immutable updates | Complex state mutations |

**Why Zustand over Redux?**
- Minimal boilerplate
- No providers needed (works outside React)
- Built-in devtools
- TypeScript-first design
- Smaller bundle size

**Why React Query?**
- Automatic caching and invalidation
- Request deduplication
- Optimistic updates
- SSE/streaming support with `useQuery` mutations
- Background refetching

### UI Components

| Package | Purpose |
|---------|---------|
| `@radix-ui/react-*` | Headless UI primitives |
| `shadcn/ui` | Pre-built Radix components |
| `tailwindcss` | Utility-first CSS |
| `class-variance-authority` | Variant styling |
| `tailwind-merge` | Class merging |
| `lucide-react` | Icon library |

**Why shadcn/ui?**
- Copy-paste components (you own the code)
- Built on Radix UI (accessible)
- Tailwind-styled
- Highly customizable
- No npm dependency lock-in

### Data Display

| Package | Purpose |
|---------|---------|
| `@tanstack/react-table` | Data tables |
| `@tanstack/react-virtual` | List virtualization |
| `lightweight-charts` | TradingView charts |
| `recharts` | General charts |

### Forms & Validation

| Package | Purpose |
|---------|---------|
| `react-hook-form` | Form state management |
| `zod` | Schema validation |
| `@hookform/resolvers` | Zod integration |

### Utilities

| Package | Purpose |
|---------|---------|
| `date-fns` | Date formatting |
| `numeral` | Number formatting |
| `react-markdown` | Markdown rendering |
| `react-syntax-highlighter` | Code highlighting |
| `dompurify` | HTML sanitization |

### Authentication

| Package | Purpose |
|---------|---------|
| `@clerk/clerk-react` | React Clerk SDK |

### Routing

| Package | Purpose |
|---------|---------|
| `react-router-dom` | Client-side routing |

---

## Project Structure

```
frontend-react/
├── public/
│   ├── favicon.ico
│   └── assets/
│       └── images/
├── src/
│   ├── main.tsx                    # Entry point
│   ├── App.tsx                     # Root component
│   ├── index.css                   # Global styles + Tailwind
│   │
│   ├── components/
│   │   ├── ui/                     # shadcn/ui components
│   │   │   ├── button.tsx
│   │   │   ├── input.tsx
│   │   │   ├── dialog.tsx
│   │   │   ├── dropdown-menu.tsx
│   │   │   ├── table.tsx
│   │   │   ├── toast.tsx
│   │   │   ├── skeleton.tsx
│   │   │   └── ...
│   │   │
│   │   ├── layout/
│   │   │   ├── AppShell.tsx
│   │   │   ├── Sidebar.tsx
│   │   │   ├── Header.tsx
│   │   │   ├── MobileNav.tsx
│   │   │   └── Footer.tsx
│   │   │
│   │   └── common/
│   │       ├── LoadingSpinner.tsx
│   │       ├── ErrorBoundary.tsx
│   │       ├── EmptyState.tsx
│   │       ├── ConfirmDialog.tsx
│   │       └── Pagination.tsx
│   │
│   ├── features/
│   │   ├── chat/
│   │   │   ├── components/
│   │   │   │   ├── ChatContainer.tsx
│   │   │   │   ├── ChatInput.tsx
│   │   │   │   ├── MessageList.tsx
│   │   │   │   ├── MessageBubble.tsx
│   │   │   │   ├── ModeSelector.tsx
│   │   │   │   ├── TypingIndicator.tsx
│   │   │   │   └── TickerAutocomplete.tsx
│   │   │   ├── hooks/
│   │   │   │   ├── useChat.ts
│   │   │   │   ├── useChatStream.ts
│   │   │   │   └── useChatHistory.ts
│   │   │   ├── types.ts
│   │   │   └── index.ts
│   │   │
│   │   ├── companies/
│   │   │   ├── components/
│   │   │   │   ├── CompanySearch.tsx
│   │   │   │   ├── CompanyCard.tsx
│   │   │   │   ├── CompanyProfile.tsx
│   │   │   │   ├── FinancialOverview.tsx
│   │   │   │   ├── SegmentChart.tsx
│   │   │   │   └── PeerComparison.tsx
│   │   │   ├── hooks/
│   │   │   │   ├── useCompanySearch.ts
│   │   │   │   ├── useCompanyProfile.ts
│   │   │   │   └── useCompanyFinancials.ts
│   │   │   ├── types.ts
│   │   │   └── index.ts
│   │   │
│   │   ├── screener/
│   │   │   ├── components/
│   │   │   │   ├── ScreenerContainer.tsx
│   │   │   │   ├── ScreenerInput.tsx
│   │   │   │   ├── ResultsTable.tsx
│   │   │   │   ├── ColumnSelector.tsx
│   │   │   │   └── SaveScreenModal.tsx
│   │   │   ├── hooks/
│   │   │   │   ├── useScreenerQuery.ts
│   │   │   │   └── useScreenerStream.ts
│   │   │   ├── types.ts
│   │   │   └── index.ts
│   │   │
│   │   ├── collections/
│   │   │   ├── components/
│   │   │   │   ├── CollectionsContainer.tsx
│   │   │   │   ├── ScreenList.tsx
│   │   │   │   ├── ScreenCard.tsx
│   │   │   │   └── ScreenActions.tsx
│   │   │   ├── hooks/
│   │   │   │   ├── useScreens.ts
│   │   │   │   └── useScreenMutations.ts
│   │   │   ├── types.ts
│   │   │   └── index.ts
│   │   │
│   │   ├── charting/
│   │   │   ├── components/
│   │   │   │   ├── ChartContainer.tsx
│   │   │   │   ├── MetricSelector.tsx
│   │   │   │   ├── CompanyMultiSelect.tsx
│   │   │   │   ├── DateRangePicker.tsx
│   │   │   │   └── ChartLegend.tsx
│   │   │   ├── hooks/
│   │   │   │   ├── useChartData.ts
│   │   │   │   └── useChartConfig.ts
│   │   │   ├── types.ts
│   │   │   └── index.ts
│   │   │
│   │   └── auth/
│   │       ├── components/
│   │       │   ├── AuthProvider.tsx
│   │       │   ├── ProtectedRoute.tsx
│   │       │   └── UserButton.tsx
│   │       ├── hooks/
│   │       │   └── useAuth.ts
│   │       └── index.ts
│   │
│   ├── hooks/
│   │   ├── useApi.ts                # Base API hook
│   │   ├── useDebounce.ts
│   │   ├── useLocalStorage.ts
│   │   ├── useMediaQuery.ts
│   │   └── useToast.ts
│   │
│   ├── lib/
│   │   ├── api.ts                   # API client (fetch wrapper)
│   │   ├── utils.ts                 # cn(), formatters
│   │   ├── constants.ts             # App constants
│   │   └── validators.ts            # Zod schemas
│   │
│   ├── stores/
│   │   ├── uiStore.ts               # UI state (sidebar, modals)
│   │   ├── chatStore.ts             # Chat messages state
│   │   └── preferencesStore.ts      # User preferences
│   │
│   ├── types/
│   │   ├── api.ts                   # API response types
│   │   ├── company.ts
│   │   ├── screener.ts
│   │   ├── chat.ts
│   │   └── common.ts
│   │
│   └── config/
│       ├── env.ts                   # Environment variables
│       └── routes.ts                # Route definitions
│
├── tailwind.config.ts
├── postcss.config.js
├── tsconfig.json
├── vite.config.ts
├── package.json
└── .env.example
```

---

## Component Architecture

### Design Principles

1. **Single Responsibility** - Each component does one thing well
2. **Composition** - Build complex UIs from simple components
3. **Props Down, Events Up** - Unidirectional data flow
4. **Colocation** - Keep related code together (hooks, types, components)
5. **Controlled Components** - Form state managed by parent

### Component Hierarchy

```
App
├── ClerkProvider
│   └── QueryClientProvider
│       └── AppShell
│           ├── Sidebar
│           │   ├── Logo
│           │   ├── NavItems
│           │   │   └── NavItem (×8)
│           │   └── UserMenu
│           │
│           ├── Header (mobile)
│           │   ├── MobileMenuButton
│           │   └── Logo
│           │
│           └── MainContent
│               └── Routes
│                   ├── /chat → ChatContainer
│                   │   ├── MessageList
│                   │   │   └── MessageBubble (×n)
│                   │   ├── ChatInput
│                   │   │   ├── ModeSelector
│                   │   │   ├── TextArea
│                   │   │   └── SendButton
│                   │   └── TickerAutocomplete
│                   │
│                   ├── /companies → CompaniesContainer
│                   │   ├── CompanySearch
│                   │   └── CompanyProfile
│                   │       ├── FinancialOverview
│                   │       ├── SegmentChart
│                   │       └── PeerComparison
│                   │
│                   ├── /screener → ScreenerContainer
│                   │   ├── ScreenerInput
│                   │   └── ResultsTable
│                   │       ├── TableHeader
│                   │       ├── TableBody (virtualized)
│                   │       └── Pagination
│                   │
│                   ├── /collections → CollectionsContainer
│                   │   └── ScreenList
│                   │       └── ScreenCard (×n)
│                   │
│                   └── /charting → ChartingContainer
│                       ├── ChartControls
│                       │   ├── MetricSelector
│                       │   ├── CompanyMultiSelect
│                       │   └── DateRangePicker
│                       └── ChartDisplay
│                           ├── Chart
│                           └── ChartLegend
```

### Example Component Implementation

```tsx
// src/features/chat/components/ChatInput.tsx
import { useState, useRef, useCallback } from 'react';
import { Send } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { ModeSelector } from './ModeSelector';
import { TickerAutocomplete } from './TickerAutocomplete';
import { useChatStore } from '@/stores/chatStore';
import { cn } from '@/lib/utils';

interface ChatInputProps {
  onSend: (message: string, mode: 'ask' | 'agent') => void;
  isLoading?: boolean;
  maxLength?: number;
}

export function ChatInput({
  onSend,
  isLoading = false,
  maxLength = 2000
}: ChatInputProps) {
  const [message, setMessage] = useState('');
  const [showAutocomplete, setShowAutocomplete] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const mode = useChatStore((state) => state.mode);
  const setMode = useChatStore((state) => state.setMode);

  const characterCount = message.length;
  const isOverLimit = characterCount > maxLength;
  const canSend = message.trim().length > 0 && !isOverLimit && !isLoading;

  const handleSend = useCallback(() => {
    if (!canSend) return;
    onSend(message.trim(), mode);
    setMessage('');
  }, [message, mode, canSend, onSend]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }, [handleSend]);

  const handleTickerSelect = useCallback((ticker: string) => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    // Insert ticker at cursor position
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const newMessage = message.slice(0, start) + ticker + message.slice(end);
    setMessage(newMessage);
    setShowAutocomplete(false);
  }, [message]);

  return (
    <div className="border-t bg-background p-4">
      <div className="relative">
        <div className="flex items-center gap-2 mb-2">
          <ModeSelector value={mode} onChange={setMode} />
        </div>

        <Textarea
          ref={textareaRef}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={mode === 'agent'
            ? "Ask complex research questions..."
            : "Ask a quick question..."
          }
          className="min-h-[80px] pr-12 resize-none"
          disabled={isLoading}
        />

        <Button
          size="icon"
          className="absolute bottom-3 right-3"
          onClick={handleSend}
          disabled={!canSend}
        >
          <Send className="h-4 w-4" />
        </Button>

        {showAutocomplete && (
          <TickerAutocomplete
            query={message}
            onSelect={handleTickerSelect}
            onClose={() => setShowAutocomplete(false)}
          />
        )}
      </div>

      <div className="flex justify-between items-center mt-2 text-xs text-muted-foreground">
        <span>Press Enter to send, Shift+Enter for new line</span>
        <span className={cn(isOverLimit && 'text-destructive')}>
          {characterCount}/{maxLength}
        </span>
      </div>
    </div>
  );
}
```

---

## State Management Strategy

### State Categories

| Category | Tool | Examples |
|----------|------|----------|
| Server State | React Query | API data, company profiles, search results |
| UI State | Zustand | Modal open/close, sidebar collapsed, active section |
| Form State | React Hook Form | Input values, validation errors |
| URL State | React Router | Current route, query params |
| Local State | useState | Component-specific temporary state |

### Zustand Store Example

```typescript
// src/stores/uiStore.ts
import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

interface UIState {
  // Sidebar
  sidebarCollapsed: boolean;
  setSidebarCollapsed: (collapsed: boolean) => void;
  toggleSidebar: () => void;

  // Active section (for mobile nav)
  activeSection: string;
  setActiveSection: (section: string) => void;

  // Modal management
  openModals: string[];
  openModal: (modalId: string) => void;
  closeModal: (modalId: string) => void;
  isModalOpen: (modalId: string) => boolean;

  // Theme
  theme: 'light' | 'dark' | 'system';
  setTheme: (theme: 'light' | 'dark' | 'system') => void;
}

export const useUIStore = create<UIState>()(
  devtools(
    persist(
      (set, get) => ({
        // Sidebar
        sidebarCollapsed: false,
        setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),
        toggleSidebar: () => set((state) => ({
          sidebarCollapsed: !state.sidebarCollapsed
        })),

        // Active section
        activeSection: 'chat',
        setActiveSection: (section) => set({ activeSection: section }),

        // Modals
        openModals: [],
        openModal: (modalId) => set((state) => ({
          openModals: [...state.openModals, modalId]
        })),
        closeModal: (modalId) => set((state) => ({
          openModals: state.openModals.filter((id) => id !== modalId)
        })),
        isModalOpen: (modalId) => get().openModals.includes(modalId),

        // Theme
        theme: 'system',
        setTheme: (theme) => set({ theme }),
      }),
      {
        name: 'stratalens-ui',
        partialize: (state) => ({
          sidebarCollapsed: state.sidebarCollapsed,
          theme: state.theme,
        }),
      }
    )
  )
);
```

### Chat Store Example

```typescript
// src/stores/chatStore.ts
import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
  error?: string;
  metadata?: {
    mode: 'ask' | 'agent';
    tables?: any[];
    charts?: any[];
  };
}

interface ChatState {
  messages: Message[];
  mode: 'ask' | 'agent';
  isLoading: boolean;
  streamingMessageId: string | null;

  // Actions
  addMessage: (message: Omit<Message, 'id' | 'timestamp'>) => string;
  updateMessage: (id: string, updates: Partial<Message>) => void;
  appendToMessage: (id: string, content: string) => void;
  removeMessage: (id: string) => void;
  clearMessages: () => void;
  setMode: (mode: 'ask' | 'agent') => void;
  setLoading: (loading: boolean) => void;
  setStreamingMessageId: (id: string | null) => void;
}

export const useChatStore = create<ChatState>()(
  immer((set, get) => ({
    messages: [],
    mode: 'ask',
    isLoading: false,
    streamingMessageId: null,

    addMessage: (message) => {
      const id = crypto.randomUUID();
      set((state) => {
        state.messages.push({
          ...message,
          id,
          timestamp: new Date(),
        });
      });
      return id;
    },

    updateMessage: (id, updates) => {
      set((state) => {
        const index = state.messages.findIndex((m) => m.id === id);
        if (index !== -1) {
          state.messages[index] = { ...state.messages[index], ...updates };
        }
      });
    },

    appendToMessage: (id, content) => {
      set((state) => {
        const index = state.messages.findIndex((m) => m.id === id);
        if (index !== -1) {
          state.messages[index].content += content;
        }
      });
    },

    removeMessage: (id) => {
      set((state) => {
        state.messages = state.messages.filter((m) => m.id !== id);
      });
    },

    clearMessages: () => set({ messages: [] }),
    setMode: (mode) => set({ mode }),
    setLoading: (loading) => set({ isLoading: loading }),
    setStreamingMessageId: (id) => set({ streamingMessageId: id }),
  }))
);
```

### React Query Usage

```typescript
// src/features/companies/hooks/useCompanySearch.ts
import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';
import type { CompanySearchResult } from '@/types/company';

export function useCompanySearch(query: string, enabled = true) {
  return useQuery({
    queryKey: ['companies', 'search', query],
    queryFn: async () => {
      if (!query || query.length < 2) return [];
      const response = await api.get<CompanySearchResult[]>(
        `/companies/search?q=${encodeURIComponent(query)}`
      );
      return response.data;
    },
    enabled: enabled && query.length >= 2,
    staleTime: 1000 * 60 * 5, // 5 minutes
    gcTime: 1000 * 60 * 30, // 30 minutes
  });
}

// src/features/companies/hooks/useCompanyProfile.ts
import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';
import type { CompanyProfile } from '@/types/company';

export function useCompanyProfile(ticker: string | null) {
  return useQuery({
    queryKey: ['companies', ticker, 'profile'],
    queryFn: async () => {
      const response = await api.get<CompanyProfile>(
        `/companies/${ticker}/profile`
      );
      return response.data;
    },
    enabled: !!ticker,
    staleTime: 1000 * 60 * 10, // 10 minutes
  });
}
```

### SSE Streaming Hook

```typescript
// src/features/chat/hooks/useChatStream.ts
import { useCallback, useRef } from 'react';
import { useChatStore } from '@/stores/chatStore';
import { useAuth } from '@clerk/clerk-react';

export function useChatStream() {
  const { getToken } = useAuth();
  const abortControllerRef = useRef<AbortController | null>(null);

  const addMessage = useChatStore((state) => state.addMessage);
  const appendToMessage = useChatStore((state) => state.appendToMessage);
  const updateMessage = useChatStore((state) => state.updateMessage);
  const setLoading = useChatStore((state) => state.setLoading);
  const setStreamingMessageId = useChatStore((state) => state.setStreamingMessageId);

  const sendMessage = useCallback(async (
    content: string,
    mode: 'ask' | 'agent'
  ) => {
    // Cancel any existing stream
    abortControllerRef.current?.abort();
    abortControllerRef.current = new AbortController();

    // Add user message
    addMessage({ role: 'user', content, metadata: { mode } });

    // Add placeholder assistant message
    const assistantMessageId = addMessage({
      role: 'assistant',
      content: '',
      isStreaming: true,
      metadata: { mode }
    });

    setLoading(true);
    setStreamingMessageId(assistantMessageId);

    try {
      const token = await getToken();

      const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({ message: content, mode }),
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error('No response body');
      }

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') continue;

            try {
              const parsed = JSON.parse(data);
              if (parsed.content) {
                appendToMessage(assistantMessageId, parsed.content);
              }
              if (parsed.tables) {
                updateMessage(assistantMessageId, {
                  metadata: { tables: parsed.tables }
                });
              }
            } catch {
              // Non-JSON content, append directly
              appendToMessage(assistantMessageId, data);
            }
          }
        }
      }

      updateMessage(assistantMessageId, { isStreaming: false });

    } catch (error) {
      if ((error as Error).name === 'AbortError') {
        updateMessage(assistantMessageId, {
          isStreaming: false,
          error: 'Message cancelled'
        });
      } else {
        updateMessage(assistantMessageId, {
          isStreaming: false,
          error: (error as Error).message
        });
      }
    } finally {
      setLoading(false);
      setStreamingMessageId(null);
    }
  }, [getToken, addMessage, appendToMessage, updateMessage, setLoading, setStreamingMessageId]);

  const cancelStream = useCallback(() => {
    abortControllerRef.current?.abort();
  }, []);

  return { sendMessage, cancelStream };
}
```

---

## API Layer & TypeScript Interfaces

### API Client

```typescript
// src/lib/api.ts
import { useAuth } from '@clerk/clerk-react';

const API_BASE_URL = import.meta.env.VITE_API_URL || '';

interface ApiResponse<T> {
  data: T;
  status: number;
}

interface ApiError {
  message: string;
  status: number;
  details?: any;
}

class ApiClient {
  private baseUrl: string;
  private getToken: (() => Promise<string | null>) | null = null;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  setTokenGetter(getter: () => Promise<string | null>) {
    this.getToken = getter;
  }

  private async getHeaders(): Promise<HeadersInit> {
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    };

    if (this.getToken) {
      const token = await this.getToken();
      if (token) {
        headers['Authorization'] = `Bearer ${token}`;
      }
    }

    return headers;
  }

  async get<T>(endpoint: string): Promise<ApiResponse<T>> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method: 'GET',
      headers: await this.getHeaders(),
    });

    if (!response.ok) {
      throw await this.handleError(response);
    }

    return {
      data: await response.json(),
      status: response.status,
    };
  }

  async post<T>(endpoint: string, body?: any): Promise<ApiResponse<T>> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method: 'POST',
      headers: await this.getHeaders(),
      body: body ? JSON.stringify(body) : undefined,
    });

    if (!response.ok) {
      throw await this.handleError(response);
    }

    return {
      data: await response.json(),
      status: response.status,
    };
  }

  async delete<T>(endpoint: string): Promise<ApiResponse<T>> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method: 'DELETE',
      headers: await this.getHeaders(),
    });

    if (!response.ok) {
      throw await this.handleError(response);
    }

    return {
      data: await response.json(),
      status: response.status,
    };
  }

  private async handleError(response: Response): Promise<ApiError> {
    let message = 'An error occurred';
    let details;

    try {
      const body = await response.json();
      message = body.detail || body.message || message;
      details = body;
    } catch {
      message = response.statusText;
    }

    return { message, status: response.status, details };
  }
}

export const api = new ApiClient(API_BASE_URL);

// Hook to initialize API with auth
export function useApiAuth() {
  const { getToken } = useAuth();

  React.useEffect(() => {
    api.setTokenGetter(getToken);
  }, [getToken]);
}
```

### Type Definitions

```typescript
// src/types/company.ts
export interface CompanySearchResult {
  ticker: string;
  name: string;
  exchange: string;
  sector?: string;
  industry?: string;
}

export interface CompanyProfile {
  ticker: string;
  name: string;
  exchange: string;
  sector: string;
  industry: string;
  description: string;
  website: string;
  employees: number;
  ceo: string;
  headquarters: {
    city: string;
    state: string;
    country: string;
  };
  marketCap: number;
  peRatio: number;
  dividendYield: number;
  fiftyTwoWeekHigh: number;
  fiftyTwoWeekLow: number;
  logo?: string;
}

export interface CompanyFinancials {
  ticker: string;
  period: string;
  revenue: number;
  netIncome: number;
  eps: number;
  grossMargin: number;
  operatingMargin: number;
  netMargin: number;
  roe: number;
  roa: number;
  debtToEquity: number;
  currentRatio: number;
}

export interface CompanySegment {
  name: string;
  revenue: number;
  percentage: number;
  growth: number;
}

// src/types/chat.ts
export type ChatMode = 'ask' | 'agent';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  mode: ChatMode;
  isStreaming?: boolean;
  error?: string;
  metadata?: ChatMessageMetadata;
}

export interface ChatMessageMetadata {
  tables?: TableData[];
  charts?: ChartData[];
  citations?: Citation[];
  processingTime?: number;
}

export interface TableData {
  columns: string[];
  rows: any[][];
  title?: string;
}

export interface ChartData {
  type: 'line' | 'bar' | 'pie';
  data: any;
  options?: any;
}

export interface Citation {
  source: string;
  url?: string;
  text: string;
}

// src/types/screener.ts
export interface ScreenerQuery {
  query: string;
  page?: number;
  pageSize?: number;
  sortBy?: string;
  sortDirection?: 'asc' | 'desc';
}

export interface ScreenerResult {
  columns: ScreenerColumn[];
  rows: ScreenerRow[];
  totalCount: number;
  page: number;
  pageSize: number;
  query: string;
}

export interface ScreenerColumn {
  key: string;
  label: string;
  type: 'string' | 'number' | 'percentage' | 'currency' | 'date';
  sortable?: boolean;
  width?: number;
}

export interface ScreenerRow {
  ticker: string;
  [key: string]: any;
}

// src/types/screen.ts
export interface Screen {
  id: string;
  name: string;
  query: string;
  description?: string;
  createdAt: Date;
  updatedAt: Date;
  resultCount?: number;
  isPublic?: boolean;
  tags?: string[];
}

export interface CreateScreenInput {
  name: string;
  query: string;
  description?: string;
  tags?: string[];
}

// src/types/charting.ts
export interface ChartRequest {
  tickers: string[];
  metrics: string[];
  startDate: string;
  endDate: string;
  normalize?: boolean;
}

export interface ChartResponse {
  data: ChartSeriesData[];
  metadata: {
    startDate: string;
    endDate: string;
    metrics: string[];
  };
}

export interface ChartSeriesData {
  ticker: string;
  metric: string;
  values: ChartDataPoint[];
}

export interface ChartDataPoint {
  date: string;
  value: number;
}

// src/types/api.ts
export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page: number;
  pageSize: number;
  hasMore: boolean;
}

export interface ApiError {
  message: string;
  code?: string;
  details?: Record<string, any>;
}
```

### Zod Validation Schemas

```typescript
// src/lib/validators.ts
import { z } from 'zod';

export const screenerQuerySchema = z.object({
  query: z.string().min(1).max(1000),
  page: z.number().int().positive().optional().default(1),
  pageSize: z.number().int().min(10).max(100).optional().default(50),
  sortBy: z.string().optional(),
  sortDirection: z.enum(['asc', 'desc']).optional().default('desc'),
});

export const createScreenSchema = z.object({
  name: z.string().min(1).max(100),
  query: z.string().min(1).max(1000),
  description: z.string().max(500).optional(),
  tags: z.array(z.string().max(50)).max(10).optional(),
});

export const chatMessageSchema = z.object({
  message: z.string().min(1).max(2000),
  mode: z.enum(['ask', 'agent']).default('ask'),
});

export const chartRequestSchema = z.object({
  tickers: z.array(z.string()).min(1).max(10),
  metrics: z.array(z.string()).min(1).max(5),
  startDate: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
  endDate: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
  normalize: z.boolean().optional().default(false),
});

export type ScreenerQueryInput = z.infer<typeof screenerQuerySchema>;
export type CreateScreenInput = z.infer<typeof createScreenSchema>;
export type ChatMessageInput = z.infer<typeof chatMessageSchema>;
export type ChartRequestInput = z.infer<typeof chartRequestSchema>;
```

---

## Redundancy Elimination Plan

### Current Redundancies Identified

Based on the analysis of app.js (10,000+ lines), the following redundancy patterns were identified:

| Pattern | Occurrences | Solution |
|---------|-------------|----------|
| Modal open/close | ~20+ modals | Single `Modal` component + Zustand |
| Character counter | 5-6 inputs | `useCharacterCount` hook |
| Loading spinners | 15+ places | `LoadingSpinner` component |
| Pagination | 4+ tables | `Pagination` component |
| API fetch | 30+ calls | React Query + `api` client |
| Error handling | 20+ try/catch | `ErrorBoundary` + toast |
| Form validation | 10+ forms | React Hook Form + Zod |
| Table rendering | 4+ tables | TanStack Table |
| Debounced search | 3+ inputs | `useDebounce` hook |
| localStorage access | 10+ places | `useLocalStorage` hook |

### Consolidation Examples

#### 1. Modal Management

**Before (app.js pattern repeated 20+ times):**
```javascript
function openCompanyModal() {
  document.getElementById('companyModal').classList.remove('hidden');
  document.body.style.overflow = 'hidden';
}

function closeCompanyModal() {
  document.getElementById('companyModal').classList.add('hidden');
  document.body.style.overflow = '';
}
```

**After (single reusable component):**
```tsx
// src/components/ui/dialog.tsx (shadcn/ui)
import * as Dialog from '@radix-ui/react-dialog';

interface ModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  children: React.ReactNode;
}

export function Modal({ open, onOpenChange, title, children }: ModalProps) {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 bg-black/50" />
        <Dialog.Content className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-background rounded-lg p-6 w-full max-w-lg">
          <Dialog.Title className="text-lg font-semibold">{title}</Dialog.Title>
          {children}
          <Dialog.Close className="absolute top-4 right-4">
            <X className="h-4 w-4" />
          </Dialog.Close>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

// Usage with Zustand
const { isModalOpen, openModal, closeModal } = useUIStore();

<Modal
  open={isModalOpen('saveScreen')}
  onOpenChange={(open) => open ? openModal('saveScreen') : closeModal('saveScreen')}
  title="Save Screen"
>
  <SaveScreenForm />
</Modal>
```

#### 2. Character Counter

**Before (duplicated 5-6 times):**
```javascript
chatInput.addEventListener('input', function() {
  const count = this.value.length;
  document.getElementById('charCount').textContent = `${count}/2000`;
  if (count > 2000) {
    document.getElementById('charCount').classList.add('text-red-500');
  } else {
    document.getElementById('charCount').classList.remove('text-red-500');
  }
});
```

**After (reusable hook):**
```tsx
// src/hooks/useCharacterCount.ts
import { useMemo } from 'react';

interface UseCharacterCountOptions {
  maxLength: number;
  warnAt?: number;
}

export function useCharacterCount(
  value: string,
  { maxLength, warnAt = maxLength * 0.9 }: UseCharacterCountOptions
) {
  return useMemo(() => ({
    count: value.length,
    remaining: maxLength - value.length,
    isOverLimit: value.length > maxLength,
    isWarning: value.length >= warnAt,
    display: `${value.length}/${maxLength}`,
  }), [value, maxLength, warnAt]);
}

// Usage
const { display, isOverLimit } = useCharacterCount(message, { maxLength: 2000 });

<span className={cn(isOverLimit && 'text-destructive')}>
  {display}
</span>
```

#### 3. Loading States

**Before (15+ variations):**
```javascript
function showLoadingSpinner(containerId) {
  document.getElementById(containerId).innerHTML = `
    <div class="flex justify-center items-center p-8">
      <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
    </div>
  `;
}
```

**After (single component with variants):**
```tsx
// src/components/common/LoadingSpinner.tsx
import { cn } from '@/lib/utils';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  className?: string;
  label?: string;
}

const sizeClasses = {
  sm: 'h-4 w-4',
  md: 'h-8 w-8',
  lg: 'h-12 w-12',
};

export function LoadingSpinner({
  size = 'md',
  className,
  label
}: LoadingSpinnerProps) {
  return (
    <div className={cn('flex flex-col items-center justify-center gap-2', className)}>
      <div
        className={cn(
          'animate-spin rounded-full border-2 border-primary border-t-transparent',
          sizeClasses[size]
        )}
      />
      {label && <span className="text-sm text-muted-foreground">{label}</span>}
    </div>
  );
}

// Full-page loading
export function PageLoader({ label = 'Loading...' }: { label?: string }) {
  return (
    <div className="flex items-center justify-center min-h-[400px]">
      <LoadingSpinner size="lg" label={label} />
    </div>
  );
}

// Skeleton for content loading
export { Skeleton } from '@/components/ui/skeleton';
```

#### 4. Pagination

**Before (4+ implementations):**
```javascript
function updatePagination(currentPage, totalPages, containerId) {
  const container = document.getElementById(containerId);
  container.innerHTML = `
    <button onclick="goToPage(${currentPage - 1})" ${currentPage === 1 ? 'disabled' : ''}>
      Previous
    </button>
    <span>Page ${currentPage} of ${totalPages}</span>
    <button onclick="goToPage(${currentPage + 1})" ${currentPage === totalPages ? 'disabled' : ''}>
      Next
    </button>
  `;
}
```

**After (single reusable component):**
```tsx
// src/components/common/Pagination.tsx
import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface PaginationProps {
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
  showFirstLast?: boolean;
  className?: string;
}

export function Pagination({
  currentPage,
  totalPages,
  onPageChange,
  showFirstLast = true,
  className,
}: PaginationProps) {
  const canGoPrevious = currentPage > 1;
  const canGoNext = currentPage < totalPages;

  return (
    <div className={cn('flex items-center gap-2', className)}>
      {showFirstLast && (
        <Button
          variant="outline"
          size="icon"
          onClick={() => onPageChange(1)}
          disabled={!canGoPrevious}
        >
          <ChevronsLeft className="h-4 w-4" />
        </Button>
      )}

      <Button
        variant="outline"
        size="icon"
        onClick={() => onPageChange(currentPage - 1)}
        disabled={!canGoPrevious}
      >
        <ChevronLeft className="h-4 w-4" />
      </Button>

      <span className="text-sm text-muted-foreground px-2">
        Page {currentPage} of {totalPages}
      </span>

      <Button
        variant="outline"
        size="icon"
        onClick={() => onPageChange(currentPage + 1)}
        disabled={!canGoNext}
      >
        <ChevronRight className="h-4 w-4" />
      </Button>

      {showFirstLast && (
        <Button
          variant="outline"
          size="icon"
          onClick={() => onPageChange(totalPages)}
          disabled={!canGoNext}
        >
          <ChevronsRight className="h-4 w-4" />
        </Button>
      )}
    </div>
  );
}
```

#### 5. Table Rendering with TanStack Table

**Before (custom table rendering repeated 4+ times):**
```javascript
function renderTable(data, columns, containerId) {
  let html = '<table class="min-w-full divide-y divide-gray-200">';
  html += '<thead><tr>';
  columns.forEach(col => {
    html += `<th onclick="sortBy('${col.key}')">${col.label}</th>`;
  });
  html += '</tr></thead><tbody>';
  data.forEach(row => {
    html += '<tr>';
    columns.forEach(col => {
      html += `<td>${formatValue(row[col.key], col.type)}</td>`;
    });
    html += '</tr>';
  });
  html += '</tbody></table>';
  document.getElementById(containerId).innerHTML = html;
}
```

**After (TanStack Table with type safety):**
```tsx
// src/components/common/DataTable.tsx
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getPaginationRowModel,
  flexRender,
  type ColumnDef,
  type SortingState,
} from '@tanstack/react-table';
import { useState } from 'react';
import { ArrowUpDown, ArrowUp, ArrowDown } from 'lucide-react';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Pagination } from './Pagination';

interface DataTableProps<TData> {
  data: TData[];
  columns: ColumnDef<TData>[];
  pageSize?: number;
  onRowClick?: (row: TData) => void;
}

export function DataTable<TData>({
  data,
  columns,
  pageSize = 20,
  onRowClick
}: DataTableProps<TData>) {
  const [sorting, setSorting] = useState<SortingState>([]);

  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    onSortingChange: setSorting,
    state: { sorting },
    initialState: { pagination: { pageSize } },
  });

  return (
    <div className="space-y-4">
      <div className="rounded-md border">
        <Table>
          <TableHeader>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <TableHead
                    key={header.id}
                    className={header.column.getCanSort() ? 'cursor-pointer select-none' : ''}
                    onClick={header.column.getToggleSortingHandler()}
                  >
                    <div className="flex items-center gap-2">
                      {flexRender(header.column.columnDef.header, header.getContext())}
                      {header.column.getCanSort() && (
                        header.column.getIsSorted() === 'asc' ? (
                          <ArrowUp className="h-4 w-4" />
                        ) : header.column.getIsSorted() === 'desc' ? (
                          <ArrowDown className="h-4 w-4" />
                        ) : (
                          <ArrowUpDown className="h-4 w-4 opacity-50" />
                        )
                      )}
                    </div>
                  </TableHead>
                ))}
              </TableRow>
            ))}
          </TableHeader>
          <TableBody>
            {table.getRowModel().rows.map((row) => (
              <TableRow
                key={row.id}
                className={onRowClick ? 'cursor-pointer hover:bg-muted' : ''}
                onClick={() => onRowClick?.(row.original)}
              >
                {row.getVisibleCells().map((cell) => (
                  <TableCell key={cell.id}>
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      <Pagination
        currentPage={table.getState().pagination.pageIndex + 1}
        totalPages={table.getPageCount()}
        onPageChange={(page) => table.setPageIndex(page - 1)}
      />
    </div>
  );
}
```

---

## Design System & Tailwind Configuration

### Tailwind Configuration

```typescript
// tailwind.config.ts
import type { Config } from 'tailwindcss';
import { fontFamily } from 'tailwindcss/defaultTheme';

export default {
  darkMode: ['class'],
  content: ['./src/**/*.{ts,tsx}'],
  theme: {
    container: {
      center: true,
      padding: '2rem',
      screens: {
        '2xl': '1400px',
      },
    },
    extend: {
      colors: {
        // Map existing CSS variables to Tailwind
        border: 'hsl(var(--border))',
        input: 'hsl(var(--input))',
        ring: 'hsl(var(--ring))',
        background: 'hsl(var(--background))',
        foreground: 'hsl(var(--foreground))',
        primary: {
          DEFAULT: 'hsl(var(--primary))',
          foreground: 'hsl(var(--primary-foreground))',
        },
        secondary: {
          DEFAULT: 'hsl(var(--secondary))',
          foreground: 'hsl(var(--secondary-foreground))',
        },
        destructive: {
          DEFAULT: 'hsl(var(--destructive))',
          foreground: 'hsl(var(--destructive-foreground))',
        },
        muted: {
          DEFAULT: 'hsl(var(--muted))',
          foreground: 'hsl(var(--muted-foreground))',
        },
        accent: {
          DEFAULT: 'hsl(var(--accent))',
          foreground: 'hsl(var(--accent-foreground))',
        },
        popover: {
          DEFAULT: 'hsl(var(--popover))',
          foreground: 'hsl(var(--popover-foreground))',
        },
        card: {
          DEFAULT: 'hsl(var(--card))',
          foreground: 'hsl(var(--card-foreground))',
        },
        // Financial-specific colors
        positive: {
          DEFAULT: 'hsl(142, 76%, 36%)',
          light: 'hsl(142, 76%, 95%)',
        },
        negative: {
          DEFAULT: 'hsl(0, 84%, 60%)',
          light: 'hsl(0, 84%, 95%)',
        },
      },
      borderRadius: {
        lg: 'var(--radius)',
        md: 'calc(var(--radius) - 2px)',
        sm: 'calc(var(--radius) - 4px)',
      },
      fontFamily: {
        sans: ['Inter var', ...fontFamily.sans],
        mono: ['JetBrains Mono', ...fontFamily.mono],
      },
      keyframes: {
        'accordion-down': {
          from: { height: '0' },
          to: { height: 'var(--radix-accordion-content-height)' },
        },
        'accordion-up': {
          from: { height: 'var(--radix-accordion-content-height)' },
          to: { height: '0' },
        },
        shimmer: {
          '100%': { transform: 'translateX(100%)' },
        },
        'pulse-subtle': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.7' },
        },
      },
      animation: {
        'accordion-down': 'accordion-down 0.2s ease-out',
        'accordion-up': 'accordion-up 0.2s ease-out',
        shimmer: 'shimmer 2s infinite',
        'pulse-subtle': 'pulse-subtle 2s ease-in-out infinite',
      },
    },
  },
  plugins: [
    require('tailwindcss-animate'),
    require('@tailwindcss/typography'),
  ],
} satisfies Config;
```

### CSS Variables (index.css)

```css
/* src/index.css */
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  :root {
    /* Light theme - matching existing styles.css */
    --background: 0 0% 100%;
    --foreground: 222.2 84% 4.9%;

    --card: 0 0% 100%;
    --card-foreground: 222.2 84% 4.9%;

    --popover: 0 0% 100%;
    --popover-foreground: 222.2 84% 4.9%;

    --primary: 221.2 83.2% 53.3%;
    --primary-foreground: 210 40% 98%;

    --secondary: 210 40% 96.1%;
    --secondary-foreground: 222.2 47.4% 11.2%;

    --muted: 210 40% 96.1%;
    --muted-foreground: 215.4 16.3% 46.9%;

    --accent: 210 40% 96.1%;
    --accent-foreground: 222.2 47.4% 11.2%;

    --destructive: 0 84.2% 60.2%;
    --destructive-foreground: 210 40% 98%;

    --border: 214.3 31.8% 91.4%;
    --input: 214.3 31.8% 91.4%;
    --ring: 221.2 83.2% 53.3%;

    --radius: 0.5rem;

    /* Sidebar */
    --sidebar-width: 280px;
    --sidebar-collapsed-width: 64px;
  }

  .dark {
    --background: 222.2 84% 4.9%;
    --foreground: 210 40% 98%;

    --card: 222.2 84% 4.9%;
    --card-foreground: 210 40% 98%;

    --popover: 222.2 84% 4.9%;
    --popover-foreground: 210 40% 98%;

    --primary: 217.2 91.2% 59.8%;
    --primary-foreground: 222.2 47.4% 11.2%;

    --secondary: 217.2 32.6% 17.5%;
    --secondary-foreground: 210 40% 98%;

    --muted: 217.2 32.6% 17.5%;
    --muted-foreground: 215 20.2% 65.1%;

    --accent: 217.2 32.6% 17.5%;
    --accent-foreground: 210 40% 98%;

    --destructive: 0 62.8% 30.6%;
    --destructive-foreground: 210 40% 98%;

    --border: 217.2 32.6% 17.5%;
    --input: 217.2 32.6% 17.5%;
    --ring: 224.3 76.3% 48%;
  }
}

@layer base {
  * {
    @apply border-border;
  }

  body {
    @apply bg-background text-foreground;
    font-feature-settings: "rlig" 1, "calt" 1;
  }
}

@layer components {
  /* Financial value styling */
  .value-positive {
    @apply text-positive;
  }

  .value-negative {
    @apply text-negative;
  }

  .value-neutral {
    @apply text-muted-foreground;
  }

  /* Scrollbar styling */
  .scrollbar-thin {
    scrollbar-width: thin;
    scrollbar-color: hsl(var(--muted)) transparent;
  }

  .scrollbar-thin::-webkit-scrollbar {
    @apply w-2;
  }

  .scrollbar-thin::-webkit-scrollbar-track {
    @apply bg-transparent;
  }

  .scrollbar-thin::-webkit-scrollbar-thumb {
    @apply bg-muted rounded-full;
  }
}
```

### Design Improvement Recommendations

Based on research into professional financial dashboard design:

#### 1. Typography Enhancement

- Use Inter or SF Pro for UI text (clear, modern)
- Use JetBrains Mono for financial data (tabular figures)
- Increase line height for better readability
- Use consistent font sizes (14px base, 12px small, 16px headings)

#### 2. Color System

- Keep green/red for positive/negative but use softer tones
- Add subtle background tints for data rows
- Use consistent hover states (subtle opacity change)
- Consider color-blind friendly alternatives (use icons too)

#### 3. Spacing & Layout

- Use 8px grid system consistently
- Add more whitespace between sections
- Use cards with subtle shadows for content grouping
- Implement consistent padding (16px, 24px, 32px)

#### 4. Data Visualization

- Use TradingView Lightweight Charts for financial data
- Implement consistent tooltip styling
- Add subtle animations for data updates
- Use skeleton loaders instead of spinners where possible

#### 5. Interaction Patterns

- Add keyboard shortcuts for power users
- Implement command palette (Cmd+K)
- Add breadcrumbs for navigation context
- Use toast notifications instead of alerts

#### 6. Mobile Responsiveness

- Collapsible sidebar on tablet
- Bottom navigation on mobile
- Swipeable modals
- Touch-friendly tap targets (44px minimum)

---

## Migration Phases

### Phase 1: Foundation (Week 1-2)

**Goal**: Set up React project with core infrastructure

| Task | Details |
|------|---------|
| Initialize Vite + React + TypeScript | `npm create vite@latest frontend-react -- --template react-ts` |
| Configure Tailwind CSS | Install and configure with shadcn/ui |
| Set up project structure | Create folder structure per spec |
| Install core dependencies | Zustand, React Query, React Router |
| Configure path aliases | `@/` for `src/` |
| Set up Clerk | `@clerk/clerk-react` |
| Create API client | Fetch wrapper with auth |
| Implement base layout | AppShell, Sidebar, Header |

**Deliverable**: Empty app shell with navigation working

### Phase 2: Core Components (Week 2-3)

**Goal**: Build reusable UI component library

| Task | Details |
|------|---------|
| Install shadcn/ui components | Button, Input, Dialog, Table, etc. |
| Create common components | LoadingSpinner, Pagination, ErrorBoundary |
| Implement theme system | Dark mode toggle |
| Create form components | With React Hook Form + Zod |
| Build modal system | Dialog + Zustand |
| Create toast system | shadcn/ui toast |

**Deliverable**: Component library documented with examples

### Phase 3: Chat Feature (Week 3-4)

**Goal**: Migrate chat functionality

| Task | Details |
|------|---------|
| Create chat store | Zustand with messages state |
| Implement SSE streaming hook | `useChatStream` |
| Build ChatInput component | With mode selector |
| Build MessageList component | With virtualization if needed |
| Build MessageBubble component | User/assistant variants |
| Implement markdown rendering | react-markdown + syntax highlighting |
| Add ticker autocomplete | Company search integration |
| Implement table rendering | In-message tables |

**Deliverable**: Fully functional chat with streaming

### Phase 4: Companies Feature (Week 4-5)

**Goal**: Migrate companies section

| Task | Details |
|------|---------|
| Create React Query hooks | Company search, profile, financials |
| Build CompanySearch component | With autocomplete |
| Build CompanyProfile component | Full company view |
| Build FinancialOverview component | Key metrics |
| Build SegmentChart component | With recharts |
| Implement comparison view | Multiple companies |

**Deliverable**: Fully functional companies section

### Phase 5: Screener Feature (Week 5-6)

**Goal**: Migrate screener with streaming results

| Task | Details |
|------|---------|
| Create screener stream hook | `useScreenerStream` |
| Build ScreenerInput component | NL query input |
| Build ResultsTable component | TanStack Table |
| Implement column visibility | User selectable |
| Implement sorting | Client + server |
| Build SaveScreenModal | Save query |
| Implement export | CSV download |

**Deliverable**: Fully functional screener with streaming

### Phase 6: Collections & Charting (Week 6-7)

**Goal**: Complete remaining features

| Task | Details |
|------|---------|
| Build ScreenList component | Saved screens |
| Implement CRUD operations | Create, update, delete |
| Build ChartContainer | Multi-company charts |
| Integrate TradingView charts | lightweight-charts |
| Build metric/company selectors | Multi-select |
| Implement date range picker | shadcn/ui date picker |

**Deliverable**: All features migrated

### Phase 7: Polish & Testing (Week 7-8)

**Goal**: Production readiness

| Task | Details |
|------|---------|
| Write unit tests | Jest + React Testing Library |
| Write E2E tests | Playwright |
| Performance optimization | Bundle analysis, lazy loading |
| Accessibility audit | WCAG compliance |
| Mobile responsiveness | Test all breakpoints |
| Documentation | Component docs, README |
| Landing page migration | Convert landing.html |

**Deliverable**: Production-ready application

### Phase 8: Deployment & Cutover (Week 8)

**Goal**: Production deployment

| Task | Details |
|------|---------|
| Set up CI/CD | GitHub Actions |
| Configure production build | Vite build optimization |
| Set up staging environment | Test deployment |
| Run parallel with old frontend | Feature flags |
| Monitor errors | Sentry or LogRocket |
| Gradual rollout | Percentage-based |
| Full cutover | Remove old frontend |

---

## Testing Strategy

### Unit Testing

```typescript
// Example: ChatInput.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ChatInput } from './ChatInput';

describe('ChatInput', () => {
  it('should render with placeholder', () => {
    render(<ChatInput onSend={jest.fn()} />);
    expect(screen.getByPlaceholderText(/ask/i)).toBeInTheDocument();
  });

  it('should show character count', async () => {
    render(<ChatInput onSend={jest.fn()} maxLength={100} />);
    const input = screen.getByRole('textbox');
    await userEvent.type(input, 'Hello world');
    expect(screen.getByText('11/100')).toBeInTheDocument();
  });

  it('should call onSend when clicking send button', async () => {
    const onSend = jest.fn();
    render(<ChatInput onSend={onSend} />);
    const input = screen.getByRole('textbox');
    await userEvent.type(input, 'Test message');
    await userEvent.click(screen.getByRole('button', { name: /send/i }));
    expect(onSend).toHaveBeenCalledWith('Test message', 'ask');
  });

  it('should disable send when over character limit', async () => {
    render(<ChatInput onSend={jest.fn()} maxLength={10} />);
    const input = screen.getByRole('textbox');
    await userEvent.type(input, 'This is too long');
    expect(screen.getByRole('button', { name: /send/i })).toBeDisabled();
  });
});
```

### Integration Testing

```typescript
// Example: Chat flow test
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ChatContainer } from './ChatContainer';

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: false } },
});

describe('Chat Flow', () => {
  it('should send message and display streaming response', async () => {
    // Mock SSE
    const mockEventSource = {
      onmessage: null,
      close: jest.fn(),
    };
    global.EventSource = jest.fn(() => mockEventSource);

    render(
      <QueryClientProvider client={queryClient}>
        <ChatContainer />
      </QueryClientProvider>
    );

    // Type and send message
    const input = screen.getByPlaceholderText(/ask/i);
    await userEvent.type(input, 'What is Apple revenue?');
    await userEvent.click(screen.getByRole('button', { name: /send/i }));

    // Verify user message appears
    expect(screen.getByText('What is Apple revenue?')).toBeInTheDocument();

    // Simulate streaming response
    mockEventSource.onmessage({ data: JSON.stringify({ content: 'Apple' }) });
    mockEventSource.onmessage({ data: JSON.stringify({ content: ' revenue' }) });
    mockEventSource.onmessage({ data: JSON.stringify({ content: ' is $394B' }) });
    mockEventSource.onmessage({ data: '[DONE]' });

    await waitFor(() => {
      expect(screen.getByText(/Apple revenue is \$394B/)).toBeInTheDocument();
    });
  });
});
```

### E2E Testing with Playwright

```typescript
// Example: e2e/chat.spec.ts
import { test, expect } from '@playwright/test';

test.describe('Chat', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/chat');
  });

  test('should send a message and receive response', async ({ page }) => {
    // Type message
    await page.getByPlaceholder(/ask/i).fill('Hello');

    // Send
    await page.getByRole('button', { name: /send/i }).click();

    // Wait for response
    await expect(page.locator('.message-assistant')).toBeVisible({ timeout: 30000 });
  });

  test('should switch between Ask and Agent modes', async ({ page }) => {
    // Open mode dropdown
    await page.getByRole('button', { name: /ask/i }).click();

    // Select Agent
    await page.getByRole('option', { name: /agent/i }).click();

    // Verify mode changed
    await expect(page.getByRole('button', { name: /agent/i })).toBeVisible();
  });
});
```

---

## Appendix: File-by-File Migration Mapping

| Current File | React Equivalent | Notes |
|--------------|-----------------|-------|
| `index.html` | `index.html` + components | Split into components |
| `app.js` | Multiple feature modules | Split by feature |
| `chat.js` | `features/chat/` | ChatContainer + hooks |
| `styles.css` | `index.css` + Tailwind | Migrate to utilities |
| `auth.js` | `features/auth/` | Use @clerk/clerk-react |
| `utils.js` | `lib/utils.ts` | Keep formatters |
| `charts.js` | `features/charting/` | Use lightweight-charts |
| `session.js` | Zustand store | useSessionStore |
| `config.js` | `config/env.ts` | Vite env vars |
| `landing.html` | Separate React app or pages | Consider Next.js |
| `landing-integration.js` | React Router state | URL params + store |

---

## Conclusion

This migration plan provides a comprehensive roadmap for converting the StrataLens frontend from vanilla JavaScript to React/TypeScript. The key benefits include:

1. **70%+ code reduction** through component reuse and eliminating redundancy
2. **Type safety** preventing runtime errors
3. **Better developer experience** with modern tooling
4. **Improved maintainability** through clear architecture
5. **Enhanced performance** with React's virtual DOM and optimizations
6. **Professional design** using Tailwind CSS and shadcn/ui

The phased approach allows for incremental migration while maintaining a working application throughout the process. Each phase produces a deliverable that can be tested and validated before moving to the next phase.

Estimated timeline: 8 weeks for complete migration with a single developer. Can be parallelized with multiple developers working on different features simultaneously after Phase 2.

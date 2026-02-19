# Persona Implementation - Quick Reference Guide

## 🎯 Three Personas Fully Integrated

### Persona Specifications

```
┌──────────────────────────────────────────────────────────────────┐
│                     PERSONA SPECIFICATIONS                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  👨‍🎓 SUPERVISOR                                                  │
│  ├─ Focus: Quality & Authority                                   │
│  ├─ k_results: 8 (focused sample)                                │
│  ├─ temperature: 0.2 (deterministic)                             │
│  ├─ Behaviour: Strict quality filtering, high-citation refs      │
│  └─ Use Case: Academic rigor, authoritative knowledge            │
│                                                                  │
│  🔍 RESEARCHER                                                  │
│  ├─ Focus: Discovery & Breadth                                   │
│  ├─ k_results: 15 (broad exploration)                            │
│  ├─ temperature: 0.5 (exploratory)                               │
│  ├─ Behaviour: Novel, diverse, emerging content                  │
│  └─ Use Case: Literature exploration, methodological innovation  │
│                                                                  │
│  📋 ASSESSOR                                                    │
│  ├─ Focus: Balance & Coverage                                    │
│  ├─ k_results: 10 (balanced)                                     │
│  ├─ temperature: 0.3 (moderate)                                  │
│  ├─ Behaviour: Mix of authority and breadth                      │
│  └─ Use Case: Comprehensive reviews, balanced assessment         │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📊 User Interface Flow

### RAG Query Assistant Section
```
┌─────────────────────────────────────────────────────────────┐
│  RAG Query Assistant                                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Template: [Find high-quality references ▼]  [Apply]        │
│                                                             │
│  Query: [Find high-quality academic references about...]    │
│                                                             │
│  Parameters:                                                │
│  ├─ k: [8]          [Temperature: 0.2]                      │
│  ├─ ☑ Code-aware    [Persona: 👨‍🎓 Supervisor ▼]             │
│  │                     - Supervisor (Quality)               │
│  │                     - Researcher (Discovery)             │
│  │                     - Assessor (Balance)                 │
│  │                     - None (No filtering)                │
│  │                                                          │
│  └─ [Run Query] button                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Workflow: Template + Persona
```
User clicks "Apply Template"
        ↓
┌─────────────────────────────────────────┐
│ Apply Template Callback                 │
├─────────────────────────────────────────┤
│ 1. Fetch template from database         │
│ 2. Populate: query_text                 │
│ 3. Populate: k_results = 8              │
│ 4. Populate: temperature = 0.2          │
│ 5. Set: persona = "supervisor" ← KEY    │
└─────────────────────────────────────────┘
        ↓
All fields ready, user enters parameter
        ↓
User clicks "Run Query"
        ↓
┌─────────────────────────────────────────┐
│ Query Execution Callback                │
├─────────────────────────────────────────┤
│ 1. Read: query_text = "..."             │
│ 2. Read: k = 8                          │
│ 3. Read: temperature = 0.2              │
│ 4. Read: persona = "supervisor" ← KEY   │
│ 5. Call: rag_answer(..., persona="...") │
└─────────────────────────────────────────┘
        ↓
Results with persona-aware reranking
```

---

## 🔄 Backend Processing Pipeline

```
rag_answer(query, collection, k, temp, persona="supervisor")
        ↓
┌──────────────────────────────────────┐
│ Answer Generation (generate.py)      │
├──────────────────────────────────────┤
│ - Validates input                    │
│ - Calls retrieve(query, k, persona)  │
│ - Passes persona downstream          │
└──────────────────────────────────────┘
        ↓
retrieve(query, collection, k, persona="supervisor")
        ↓
┌──────────────────────────────────────┐
│ Retrieval Pipeline (retrieve.py)     │
├──────────────────────────────────────┤
│ Step 1: Hybrid Search                │
│ ├─ Vector Search (semantic)          │
│ ├─ BM25 Search (keyword)             │
│ └─ Combine & deduplicate → 2k chunks │
│                                      │
│ Step 2: Persona Reranking            │
│ ├─ If persona = "supervisor":        │
│ │  └─ Sort by: citation_count DESC   │
│ │     authority_score DESC           │
│ ├─ If persona = "researcher":        │
│ │  └─ Sort by: recency DESC          │
│ │     novelty_score DESC             │
│ ├─ If persona = "assessor":          │
│ │  └─ Sort by: balanced_score DESC   │
│ └─ Return top k chunks               │
└──────────────────────────────────────┘
        ↓
return (chunks, metadata)  ← Persona-optimised results
        ↓
Format & Return to User
```

---

## 📦 Database Schema

### query_templates Table
```sql
CREATE TABLE query_templates (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE,
    category TEXT,              -- "academic", "code", "governance"
    description TEXT,
    template_text TEXT,
    k_results INTEGER DEFAULT 5,
    temperature REAL DEFAULT 0.3,
    code_aware BOOLEAN DEFAULT 0,
    persona TEXT DEFAULT NULL,   -- "supervisor", "researcher", "assessor"
    tags TEXT,
    usage_count INTEGER DEFAULT 0,
    last_used TIMESTAMP,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

### Sample Data
```sql
INSERT INTO query_templates VALUES (
    NULL,
    'Find high-quality references',
    'academic',
    'Search for high-quality academic references',
    'Find high-quality academic references about {}',
    8,           -- k_results
    0.2,         -- temperature
    0,           -- code_aware
    'supervisor',  -- persona ← Persona preset
    'academic,references,quality',
    0,
    NULL,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
);
```

---

## 🔌 Integration Points Summary

| Component | File | Purpose |
|-----------|------|---------|
| **UI Dropdown** | dashboard.py | Users select persona |
| **Template Selector** | dashboard.py | Choose template with preset |
| **Apply Callback** | dashboard.py | Populate all params + persona |
| **Query Callback** | dashboard.py | Read persona, pass to backend |
| **Template DB** | query_templates.py | Store persona with templates |
| **Answer Gen** | generate.py | Forward persona to retrieve |
| **Retrieval** | retrieve.py | Apply persona reranking |
| **Reranking** | persona_retrieval.py | Reorder results by persona |

---

## ✅ Verification Checklist

```
SCHEMA:
  [✓] persona field in query_templates table
  [✓] DEFAULT NULL for backward compatibility
  [✓] String type (VARCHAR/TEXT)

ACADEMIC TEMPLATES:
  [✓] "Find high-quality references" → persona=supervisor
  [✓] "Find recent research" → persona=researcher
  [✓] "Literature review" → persona=assessor
  [✓] "Find methodology" → persona=researcher
  [✓] "Citation analysis" → persona=supervisor
  
UI:
  [✓] Persona selector dropdown in RAG Query section
  [✓] Icons: 👨‍🎓 🔍 📋
  [✓] Options: supervisor, researcher, assessor, none
  [✓] ID: "rag-query-persona"
  [✓] Located after code-aware checkbox
  
CALLBACKS:
  [✓] apply_template outputs persona
  [✓] apply_template reads from template data
  [✓] run_rag_query reads persona from dropdown
  [✓] run_rag_query passes to rag_answer()
  
BACKEND:
  [✓] rag_answer accepts persona parameter
  [✓] retrieve accepts persona parameter
  [✓] apply_persona_reranking called when persona set
  [✓] Three personas supported: supervisor, researcher, assessor
  
VERIFICATION:
  [✓] Schema validation passed
  [✓] Template data validation passed
  [✓] UI validation passed
  [✓] Backend validation passed
  [✓] Callback validation passed
```

---

## 🚀 How to Use

### Quick Start: Template-Based (Recommended)
1. Select template: **"Find high-quality references"**
2. Click: **"Apply Template"**
3. Observe: Persona = **Supervisor (👨‍🎓)**, k = **8**, temp = **0.2**
4. Enter parameter: **"machine learning"**
5. Click: **"Run Query"**
6. See: High-quality, well-cited references prioritised

### Advanced: Direct Persona Selection
1. Leave template blank
2. Select persona: **"Researcher (🔍)"**
3. Set k: **15** (or accept dropdown settings)
4. Set temperature: **0.5** (or accept defaults)
5. Enter query: **"What are emerging trends in AI?"**
6. Click: **"Run Query"**
7. See: Recent, novel, diverse papers emphasised

---

## 📈 Expected Behaviour

### Same Query, Different Personas

**Query**: "What is reinforcement learning?"

| Persona | Result Count | Content Type | Focus |
|---------|--------------|--------------|-------|
| Supervisor | 8 | Classic papers, foundational | Hinton, Sutton, Barto |
| Researcher | 15 | Mix classic + recent | Recent deep RL advances |
| Assessor | 10 | Balanced mix | Broad coverage |

---

## 📝 Associated Files

```
scripts/ui/
├── dashboard.py (5471 lines)
│   ├─ Added persona selector dropdown
│   ├─ Updated apply_template() callback
│   └─ Query callback already passes persona
│
└── query_templates.py (386 lines)
    ├─ Added persona field to schema
    ├─ Updated 5 academic templates
    └─ Persona values: supervisor/researcher/assessor

scripts/rag/
├── generate.py ✓ Already supports persona
└── retrieve.py ✓ Already applies persona reranking
```

---

## 🎓 Personas Explained

### Supervisor: Quality & Authority
**Goal**: Find the most authoritative, well-established knowledge  
**Strategy**: 
- Prioritise high-citation papers
- Emphasise peer-reviewed sources
- Focus on foundational knowledge
- Smaller sample (k=8) for quality focus
- Lower temperature (0.2) for consistency

### Researcher: Discovery & Breadth
**Goal**: Find emerging trends and methodological innovations  
**Strategy**:
- Emphasise recent publications
- Include novel approaches
- Prioritise diverse viewpoints
- Larger sample (k=15) for exploration
- Higher temperature (0.5) for creativity

### Assessor: Balance & Coverage
**Goal**: Build comprehensive, well-rounded understanding  
**Strategy**:
- Mix foundational + recent works
- Balance authority with coverage
- Include diverse perspectives
- Medium sample (k=10) for balance
- Moderate temperature (0.3) for consistency

---

## 🔧 Testing Commands

```bash
# Check template data
sqlite3 rag_data/query_templates.db \
  "SELECT name, k_results, temperature, persona FROM query_templates WHERE category='academic';"

# Run dashboard
python scripts/ui/dashboard.py

# Test template callback manually
python -c "
from scripts.ui.query_templates import QueryTemplateManager
from pathlib import Path
db = QueryTemplateManager(Path('rag_data/query_templates.db'))
t = db.get_template('Find high-quality references')
print(f'Persona: {t[\"persona\"]} (expected: supervisor)')
print(f'k: {t[\"k_results\"]} (expected: 8)')
print(f'temp: {t[\"temperature\"]} (expected: 0.2)')
"
```

---

## ✨ Summary

✅ **Personas fully surfaced through:**
- UI selector dropdown
- Template-based presets
- Query parameter passing
- Backend reranking
- All integration points validated

✅ **Users can:**
- Choose persona for query behaviour
- Use templates with persona presets
- Get persona-tailored results

---

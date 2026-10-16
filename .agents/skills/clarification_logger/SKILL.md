---
name: clarification-logger
description: Automatically captures, summarizes, and logs conceptual, physical, and methodological clarifications into clarification.md, maintaining a structured, hyperlinked table of contents and standardized entry formatting.
---

# Clarification Logger Skill

## Overview
This skill automates the capture, synthesis, and organization of scientific, mathematical, and algorithmic clarifications requested by the author (Dr. Atilio Grondona). It ensures that all deep conceptual explanations (such as "What is an ablation matrix?", "Why Sobolev loss?", "How does Planck's law constrain the network?") are permanently archived into [`clarification.md`](file:///c:/Users/Atilio/Desktop/PROJETOS/AITIR/clarification.md).

---

## When to Activate
Activate this skill whenever the user asks for a conceptual, mathematical, physical, or architectural explanation regarding the project, including prompts such as:
* *"Can you explain to me what [concept] is and why we are using it?"*
* *"What is the difference between [method A] and [method B]?"*
* *"Why did we choose this parameter / loss function / activation?"*
* *"How will Reviewer #2 evaluate this choice?"*

---

## Logging Procedure & Workflow

Whenever a clarification is requested and answered in the conversation:

### Step 1: Formulate the Response to the User
Provide the immediate answer in the conversation adhering strictly to the **English Feedback Loop (4-part format)** and the **Senior Peer-to-Peer / Q1 Reviewer Persona**.

### Step 2: Extract & Structure the Knowledge Entry
Structure the explanation into the following standardized schema:

```markdown
## [N]. [Title: Core Concept & Specific Application]
* **Date Added:** YYYY-MM-DD
* **Category:** [Physical Modeling | Neural Architecture | Remote Sensing Theory | Reviewer #2 Defense | Data Engineering]
* **Related Code & Artifacts:** [Clickable file links with file:///]

### ❓ Inquired Question
> "[Exact user question with grammar corrections applied]"

---

### 💡 Executive Summary
[Concise 2–3 sentence takeaway defining what it is and the primary practical benefit for IATIR]

---

### 🔬 Full Theoretical & Editorial Explanation

#### 1. Conceptual Definition & Origin
[Formal mathematical / physical definition, etymology, and baseline context]

#### 2. The "Reviewer #2" Reality: The Potential Research Trap
[Why top-tier Q1 journals scrutinize this aspect, and what failure modes it prevents]

#### 3. Concrete Application in the IATIR Project
[Exact mechanism in our PyTorch code, loss formulation, or sensor validation]

---

### 📌 Impact on Manuscript & Implementation
* **Placement:** [Specific section in manuscript/main.tex]
* **Key Takeaway for Reviewers:** [The decisive conclusion defending our novelty]
```

### Step 3: Update `clarification.md`
1. Read the current contents of [`clarification.md`](file:///c:/Users/Atilio/Desktop/PROJETOS/AITIR/clarification.md).
2. Determine the next sequential index `N` (e.g., `2`, `3`, `4`...).
3. Add a new hyperlinked entry to the **Table of Contents** at the top of the document:
   ```markdown
   N. [[Title: Core Concept & Specific Application]](#n-anchor-slug)
   ```
4. Append the formatted clarification section at the end of the file.
5. Ensure all file paths use clickable markdown links with the `file:///` URI scheme.

---

## Quality Checklist
- [ ] Table of Contents is synchronized with exact anchor links.
- [ ] All equations are formatted in standard GitHub/LaTeX math syntax (`$...$` and `$$...$$`).
- [ ] Links to workspace files use absolute `file:///` URIs.
- [ ] Tone remains peer-to-peer, rigorous, and directly grounded in the project codebase.

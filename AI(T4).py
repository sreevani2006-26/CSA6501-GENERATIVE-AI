# Experiment 4
# Design Zero-shot, One-shot, and Few-shot prompts
# to generate a professional leave email due to illness.

# -----------------------------
# Zero-shot Prompt
# -----------------------------
zero_shot = """
Write a professional email requesting leave due to illness.

Instructions:
- Mention the reason for leave.
- Be polite and professional.
- Request leave for one day.
- End with a thank-you note.
"""

# -----------------------------
# One-shot Prompt
# -----------------------------
one_shot = """
Example:

Write a professional email requesting permission to attend a family function.

Email:
Subject: Leave Request for Family Function

Dear Manager,

I would like to request leave for one day as I need to attend a family function. I have completed my pending work and will resume my duties the following day.

Thank you for your understanding.

Sincerely,
ABC

Now write a similar professional email requesting leave due to illness.
"""

# -----------------------------
# Few-shot Prompt
# -----------------------------
few_shot = """
Example 1

Subject: Leave Request for Personal Work

Dear Manager,

I request leave for one day due to personal work. I have completed my assigned tasks and will return to work tomorrow.

Thank you for your support.

Sincerely,
ABC

Example 2

Subject: Leave Request for Medical Appointment

Dear Manager,

I am writing to request leave for one day as I have a scheduled medical appointment. I will complete any pending work upon my return.

Thank you for your understanding.

Sincerely,
ABC

Now write a professional email requesting leave due to illness.

Instructions:
- Include a subject.
- Use polite and formal language.
- Mention illness as the reason.
- Request one day's leave.
- End with a thank-you note and signature.
"""

print("=" * 60)
print("ZERO-SHOT PROMPT")
print("=" * 60)
print(zero_shot)

print("\n" + "=" * 60)
print("ONE-SHOT PROMPT")
print("=" * 60)
print(one_shot)

print("\n" + "=" * 60)
print("FEW-SHOT PROMPT")
print("=" * 60)
print(few_shot)

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
print("1. Zero-shot : Professional tone but may miss proper formatting.")
print("2. One-shot  : Better grammar and email structure due to one example.")
print("3. Few-shot  : Best tone, grammar, formatting, and completeness because multiple examples guide the model.")

print("\nConclusion:")
print("Few-shot prompting generates the most professional and complete email.")

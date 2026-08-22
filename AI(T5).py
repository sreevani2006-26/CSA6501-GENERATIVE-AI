# Experiment 5
# Design Zero-shot, One-shot, and Few-shot prompts
# to generate a promotional social media post for an AI Workshop.

# -----------------------------
# Zero-shot Prompt
# -----------------------------
zero_shot = """
Write a promotional social media post for an AI Workshop.

Instructions:
- Use an engaging and professional tone.
- Highlight the benefits of attending the workshop.
- Include a call-to-action.
- Keep the post within 100 words.
"""

# -----------------------------
# One-shot Prompt
# -----------------------------
one_shot = """
Example:

Topic: Web Development Workshop

Social Media Post:
🚀 Join our Web Development Workshop and learn HTML, CSS, JavaScript, and modern web technologies from industry experts. Gain hands-on experience, earn a certificate, and build real-world projects. Register today and kick-start your web development journey!

Now write a similar promotional social media post for an AI Workshop.
"""

# -----------------------------
# Few-shot Prompt
# -----------------------------
few_shot = """
Example 1

Topic: Python Programming Workshop

Post:
🐍 Learn Python from scratch with hands-on coding sessions. Build exciting projects, improve your programming skills, and earn a certificate. Limited seats available. Register now!

Example 2

Topic: Cyber Security Workshop

Post:
🔒 Protect the digital world by joining our Cyber Security Workshop. Learn ethical hacking, network security, and cyber defense techniques from experts. Enroll today and secure your future!

Now write a promotional social media post for an AI Workshop.

Instructions:
- Keep it under 100 words.
- Use an attractive opening.
- Mention AI concepts, hands-on learning, expert guidance, and certificate.
- End with a strong call-to-action.
- Add relevant hashtags.
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
print("1. Zero-shot : Good promotional content but may miss hashtags or formatting.")
print("2. One-shot  : Better structure and engaging style due to one example.")
print("3. Few-shot  : Best creativity, readability, and promotional impact because multiple examples guide the model.")

print("\nConclusion:")
print("Few-shot prompting generates the most engaging and professional social media post.")

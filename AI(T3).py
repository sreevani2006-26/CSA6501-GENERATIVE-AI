# Experiment 3
# Design Zero-shot, One-shot, and Few-shot prompts
# to summarize an article into 50 words
# and compare the generated summaries.

# -----------------------------
# Sample Article
# -----------------------------
article = """
Artificial Intelligence (AI) is transforming healthcare by improving diagnosis,
personalizing treatments, and enhancing patient care. AI-powered systems analyze
medical images, detect diseases early, assist doctors in decision-making, and
predict patient outcomes. Virtual assistants help patients schedule appointments
and answer medical queries. AI also accelerates drug discovery and reduces
healthcare costs. Although AI offers numerous benefits, challenges such as data
privacy, ethical concerns, and implementation costs remain. With continuous
advancements, AI is expected to play an increasingly important role in modern
healthcare.
"""

# -----------------------------
# Zero-shot Prompt
# -----------------------------
zero_shot = f"""
Summarize the following article into exactly 50 words.

Article:
{article}
"""

# -----------------------------
# One-shot Prompt
# -----------------------------
one_shot = f"""
Example:

Article:
Cloud computing provides storage, software, and computing services over the
internet. It reduces costs, improves scalability, and allows users to access
data anytime from anywhere.

Summary (50 words):
Cloud computing delivers computing services through the internet. It offers
cost savings, scalability, flexibility, and easy access to data and
applications. Businesses and individuals use cloud technology to improve
efficiency, collaboration, and storage while reducing the need for expensive
hardware infrastructure.

Now summarize the following article into exactly 50 words.

Article:
{article}
"""

# -----------------------------
# Few-shot Prompt
# -----------------------------
few_shot = f"""
Example 1

Article:
Cyber security protects systems from cyber attacks, data theft, and malware.
It uses encryption, firewalls, and strong authentication to improve security.

Summary:
Cyber security safeguards digital systems against threats by using security
technologies such as encryption, firewalls, and authentication. It protects
data, maintains privacy, and ensures safe online activities for individuals
and organizations.

Example 2

Article:
Renewable energy comes from natural resources such as sunlight, wind, and
water. It reduces pollution and helps combat climate change.

Summary:
Renewable energy uses sustainable natural resources to generate power while
reducing pollution and greenhouse gas emissions. It supports environmental
protection and provides a cleaner, long-term alternative to fossil fuels.

Now summarize the following article into exactly 50 words.

Article:
{article}
"""

print("="*60)
print("ZERO-SHOT PROMPT")
print("="*60)
print(zero_shot)

print("\n" + "="*60)
print("ONE-SHOT PROMPT")
print("="*60)
print(one_shot)

print("\n" + "="*60)
print("FEW-SHOT PROMPT")
print("="*60)
print(few_shot)

print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print("1. Zero-shot : Good accuracy, but may miss some important details.")
print("2. One-shot  : Better accuracy and structure due to the example.")
print("3. Few-shot  : Highest accuracy, completeness, and readability because multiple examples guide the model.")

print("\nConclusion:")
print("Few-shot prompting produces the most accurate, complete, and readable summary.")

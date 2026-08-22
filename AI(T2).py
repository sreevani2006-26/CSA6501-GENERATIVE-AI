# Experiment 2
# Design Zero-shot, One-shot, and Few-shot prompts
# to generate a 200-word blog on
# "Applications of Artificial Intelligence in Healthcare"

# -------------------------
# Zero-shot Prompt
# -------------------------
zero_shot = """
Write a 200-word blog on the topic
'Applications of Artificial Intelligence in Healthcare'.

Instructions:
- Use simple and clear language.
- Explain how AI is used in healthcare.
- Mention at least four applications.
- Include a short conclusion.
"""

# -------------------------
# One-shot Prompt
# -------------------------
one_shot = """
Example:

Topic: Benefits of Online Learning

Blog:
Online learning has transformed education by making quality learning accessible to everyone. Students can attend classes from anywhere, access recorded lectures, and learn at their own pace. It saves travel time, reduces costs, and offers flexible schedules. Interactive tools and digital resources improve engagement and understanding. Online learning also supports lifelong education by allowing professionals to upgrade their skills. Although internet connectivity can sometimes be a challenge, the advantages outweigh the disadvantages. Overall, online learning has become an effective and convenient way to gain knowledge in today's digital world.

Now write a similar 200-word blog on:
'Applications of Artificial Intelligence in Healthcare'.
"""

# -------------------------
# Few-shot Prompt
# -------------------------
few_shot = """
Example 1:

Topic: Importance of Cyber Security

Blog:
Cyber security protects computers, networks, and data from cyber attacks. It helps prevent data theft, financial loss, and privacy breaches. Strong passwords, encryption, and antivirus software improve security. As technology grows, cyber security has become essential for individuals and organisations.

Example 2:

Topic: Cloud Computing

Blog:
Cloud computing provides storage, software, and computing services over the internet. It reduces hardware costs, improves scalability, and allows users to access data anytime. Many businesses use cloud computing to increase efficiency and collaboration.

Now write a 200-word blog on:
'Applications of Artificial Intelligence in Healthcare'.

Instructions:
- Keep the blog around 200 words.
- Explain AI applications in diagnosis, medical imaging, drug discovery, patient monitoring, and virtual assistants.
- End with a positive conclusion.
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

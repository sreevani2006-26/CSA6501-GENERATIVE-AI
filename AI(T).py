# Zero-shot Prompt
zero_shot = """
Generate a product description for a Smart Fitness Watch.
Mention its features, benefits, and target audience.
"""

# One-shot Prompt
one_shot = """
Example:
Product: Wireless Earbuds
Description:
Wireless Earbuds provide high-quality sound, noise cancellation,
and long battery life for music lovers.

Now generate a similar description for:
Product: Smart Fitness Watch
"""

# Few-shot Prompt
few_shot = """
Example 1:
Product: Wireless Earbuds
Description:
Wireless Earbuds offer crystal-clear sound, noise cancellation,
and long battery life.

Example 2:
Product: Smart Water Bottle
Description:
A Smart Water Bottle tracks water intake, reminds users to drink,
and syncs with a mobile app.

Now generate a description for:
Product: Smart Fitness Watch
"""

print("ZERO-SHOT PROMPT")
print(zero_shot)

print("\nONE-SHOT PROMPT")
print(one_shot)

print("\nFEW-SHOT PROMPT")
print(few_shot)

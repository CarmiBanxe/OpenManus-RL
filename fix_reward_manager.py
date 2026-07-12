# Read the file
with open('./openmanus_rl/environments/env_package/tool_use/reward_manager.py', 'r') as f:
    content = f.read()

# Replace the verification call
old_code = '''        verification = self.llm_engine(query_prompt, response_format=AnswerVerification)

        analysis = verification.analysis.strip()
        is_correct = verification.true_false'''

new_code = '''        verification_text = self.llm_engine(query_prompt)
        
        # Parse the response manually
        analysis = ""
        is_correct = False
        
        # Extract analysis
        analysis_match = re.search(r"<analysis>(.*?)</analysis>", verification_text, re.DOTALL)
        if analysis_match:
            analysis = analysis_match.group(1).strip()
        
        # Extract true_false
        tf_match = re.search(r"<true_false>(.*?)</true_false>", verification_text, re.DOTALL)
        if tf_match:
            tf_value = tf_match.group(1).strip().lower()
            is_correct = tf_value in ("true", "yes", "1")'''

content = content.replace(old_code, new_code)

# Write the modified content back to the file
with open('./openmanus_rl/environments/env_package/tool_use/reward_manager.py', 'w') as f:
    f.write(content)

print('File updated successfully')

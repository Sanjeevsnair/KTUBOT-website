def remove_underscores(text):
    return text.replace('_', '')

# Example usage
input_text = "This_is_an_example_sentence"
output_text = remove_underscores(input_text)
print(output_text)  # Output: Thisisanexamplesentence

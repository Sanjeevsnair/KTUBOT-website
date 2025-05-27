function removeUnderscores(text) {
  return text.replace(/_/g, ' ');
}

// Example usage
const input = "This_is_an_example_sentence";
const output = removeUnderscores(input);
console.log(output); // Output: Thisisanexamplesentence

function normalizeSubject(subject) {
  return subject.toLowerCase().replace(/[_\s]/g, '').trim();
}


const API_BASE_URL = 'https://ktubot-website.onrender.com/api';


async function getSubjects(department, semester, mySubject) {
  const response = await fetch(`${API_BASE_URL}/subjects`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json'
    },
    body: JSON.stringify({
      department: department,
      semester: semester
    })
  });

  if (!response.ok) {
    const error = await response.json();
    console.error('Error:', error);
    return null;
  }

  const subjects = await response.json(); // e.g., ["Data Structures", "OPERATING_SYSTEMS"]
  console.log('Subjects from API:', subjects);

  // Normalize and find match
  const normalizedMySubject = normalizeSubject(mySubject);
  const matched = subjects.find(apiSubject =>
    normalizeSubject(apiSubject) === normalizedMySubject
  );

  if (matched) {
    console.log('Matched Subject:', matched);
    return matched;
  } else {
    console.warn('No match found for:', mySubject);
    return null;
  }
}

// Example usage
getSubjects('CSE', 'Semester 5', 'Microprocessors And Microcontrollers');  // will match "OPERATING_SYSTEMS"

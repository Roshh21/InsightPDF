const BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function uploadPDF(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${BASE_URL}/upload`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error("Upload failed");
  }

  return res.json(); // { docId, summary: { docId, documentType, sections } }
}

export async function askQuestion(docId, question) {
  const res = await fetch(`${BASE_URL}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ doc_id: docId, question }),
  });

  if (!res.ok) {
    throw new Error("Query failed");
  }

  return res.json(); // { text }
}

export async function generateQuiz(docId, numQuestions = 5) {
  const res = await fetch(`${BASE_URL}/quiz`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      doc_id: docId,
      num_questions: numQuestions,
    }),
  });

  if (!res.ok) {
    throw new Error("Quiz failed");
  }

  return res.json(); // { questions: [...] }
}

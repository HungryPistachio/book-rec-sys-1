window.addEventListener("DOMContentLoaded", initTabs);
document.querySelectorAll('.get-recommendations-button').forEach(button => {
  button.addEventListener('click', async () => {
    const tab = button.dataset.tab;
    await getRecommendations(tab);
  });
});
function initTabs() {
  const tabButtons = document.querySelectorAll(".tab");
  showTabContent("noXAI");
  document.querySelector('.tab[data-tab="noXAI"]').classList.add("active");
  tabButtons.forEach(button => button.addEventListener("click", () => handleTabClick(button.dataset.tab)));
}

function handleTabClick(tabId) {
  document.querySelectorAll(".tab").forEach(button => button.classList.remove("active"));
  document.querySelector(`.tab[data-tab="${tabId}"]`).classList.add("active");
  showTabContent(tabId);
}

function showTabContent(tabId) {
  document.querySelectorAll(".tab-content").forEach(content => {
    content.style.display = content.id === tabId ? "block" : "none";
  });
}

const stopwords = new Set([
                            "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
                            "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
                            "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
                            "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
                            "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
                            "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up",
                            "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there",
                            "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some",
                            "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will",
                            "just", "don", "should", "now", "12th", "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "11th",
                            "book", "writing", "novel", "publishing", "published", "international", "bestseller"
                          ]);

function cleanDescription(description) {
  // Regular expression to remove punctuation and numbers
  const words = description
      .replace(/[0-9.,/#!$%^&*;:{}=_`~()]/g, "")
      .toLowerCase()
      .split(" ");

  // Filter out stopwords and ensure words are alphanumeric only
  const cleanedWords = words.filter(word =>
                                        word && !stopwords.has(word) && /^[a-zA-Z]+$/.test(word)
  );

  // Join the cleaned words into a cleaned description string
  return cleanedWords.join(" ");
}

function capitalize(word) {
  return word.charAt(0).toUpperCase() + word.slice(1);
}

function normalizeAuthorName(name) {
  return name.replace(/[^\w]/g, '').toLowerCase().trim(); // Remove punctuation and spaces, convert to lowercase
}

function normalizeTitle(title) {
  return title.replace(/[^\w\s]/g, '').toLowerCase().trim(); // Remove punctuation, convert to lowercase, trim spaces
}
function cosineSimilarity(vecA, vecB) {
  const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
  const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  return dotProduct / (magnitudeA * magnitudeB);
}

function validateRecommendations(recommendations) {
  return recommendations.every(rec =>
                                   rec &&
                                   typeof rec.title === 'string' &&
                                   Array.isArray(rec.vectorized_descriptions) &&
                                   Array.isArray(rec.feature_names) &&
                                   rec.vectorized_descriptions.length === rec.feature_names.length
  );
}

async function getRecommendations(tab) {
  console.log(`Fetching recommendations for tab: ${tab}`);

  const inputId = `bookTitle${capitalize(tab)}`;
  const bookTitle = document.getElementById(inputId).value.toLowerCase();
  const recommendationsId = `recommendations${capitalize(tab)}`;
  const recommendationsElement = document.getElementById(recommendationsId);
  recommendationsElement.innerHTML = '';

  const errorMessageElement = document.getElementById(`errorMessage${capitalize(tab)}`);
  const includeSameAuthor = document.getElementById(`includeSameAuthor${capitalize(tab)}`).checked;
  const googleApiKey = 'AIzaSyDQTQy3LT7BOO_LLClbWuEvqiPbUbWWKBs';
  const firstApiUrl = `https://www.googleapis.com/books/v1/volumes?q=${encodeURIComponent(bookTitle)}&maxResults=1&key=${googleApiKey}`;

  let selectedBook;
  try {
    // Step 1: Fetch input book details
    const response = await fetch(firstApiUrl);
    const data = await response.json();
    if (!data.items || data.items.length === 0) {
      throw new Error("No book found for the provided title.");
    }
    selectedBook = {
      title: data.items[0].volumeInfo.title || "Unknown Title",
      authors: data.items[0].volumeInfo.authors || [],
      description: cleanDescription(data.items[0].volumeInfo.description || "No description available"),
      link: data.items[0].volumeInfo.infoLink || "#"
    };
    console.log("Selected Book:", selectedBook);
  } catch (error) {
    console.error("Error fetching input book:", error);
    errorMessageElement.textContent = "Error fetching input book. Please try again.";
    errorMessageElement.style.display = "block";
    return;
  }

  const inputTitleNormalized = normalizeTitle(selectedBook.title);

  // Step 2: Generate keywords from the book description
  let keywords;
  try {
    const response = await fetch("/generate-keywords", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ description: selectedBook.description })
    });
    const result = await response.json();
    keywords = result.keywords;
    console.log("Generated Keywords:", keywords);
  } catch (error) {
    console.error("Error generating keywords:", error);
    errorMessageElement.textContent = "Error processing the book description. Please try again.";
    errorMessageElement.style.display = "block";
    return;
  }

  // Step 3: Fetch related books based on keywords
  const secondApiUrl = `https://www.googleapis.com/books/v1/volumes?q=${encodeURIComponent(keywords.join(" "))}&maxResults=40&key=${googleApiKey}`;
  let relatedBooks = [];
  try {
    const response = await fetch(secondApiUrl);
    const data = await response.json();

    // Step 4: Process fetched books
    relatedBooks = (data.items || []).map(book => ({
      title: book.volumeInfo.title || "Unknown Title",
      authors: book.volumeInfo.authors || [],
      description: cleanDescription(book.volumeInfo.description || "No description available"),
      link: book.volumeInfo.infoLink || "#"
    })).filter(book => book.description && book.description.length > 50);

    console.log("Related Books (Pre-Vectorization):", relatedBooks);
  } catch (error) {
    console.error("Error fetching related books:", error);
    errorMessageElement.textContent = "Error fetching related books. Please try again.";
    errorMessageElement.style.display = "block";
    return;
  }

  // Step 5: Vectorize descriptions of the related books
  let recommendations = [];
  try {
    const descriptions = relatedBooks.map(book => book.description);
    const vectorizeResponse = await fetch("/vectorize-descriptions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ descriptions })
    });

    if (!vectorizeResponse.ok) {
      throw new Error("Error during vectorization of related books");
    }

    const { tfidf_matrix, feature_names, vectorized_descriptions } = await vectorizeResponse.json();
    console.log("Vectorization Results:", { tfidf_matrix, feature_names, vectorized_descriptions });

    // Step 6: Compute similarity and filter recommendations
    const similarityScores = relatedBooks.map((book, index) => {
      const bookVector = tfidf_matrix[index];
      const similarity = cosineSimilarity(vectorized_descriptions[0], bookVector);
      return {
        book: {
          ...book,
          vectorized_descriptions: tfidf_matrix[index],
          feature_names
        },
        similarity
      };
    });

    similarityScores.sort((a, b) => b.similarity - a.similarity);

    // Step 7: Apply filters
    const addedTitles = new Set();
    recommendations = similarityScores
        .filter(({ book }) => {
          const normalizedTitle = normalizeTitle(book.title);

          // Exclude input book
          if (normalizedTitle === inputTitleNormalized) return false;

          // Avoid duplicates
          if (addedTitles.has(normalizedTitle)) return false;

          // Exclude books by the same author (if checkbox is unchecked)
          if (!includeSameAuthor && book.authors.some(author =>
                                                          selectedBook.authors.some(selectedAuthor =>
                                                                                        normalizeAuthorName(author) === normalizeAuthorName(selectedAuthor)
                                                          )
          )) return false;

          // Add title to the set of added titles
          addedTitles.add(normalizedTitle);
          return true;
        })
        .slice(0, 5) // Limit to top 5 based on similarity
        .map(entry => entry.book);

    console.log("Final Recommendations (Filtered by Title and Author):", recommendations);
  } catch (error) {
    console.error("Error processing related books:", error);
    errorMessageElement.textContent = "Error processing related books. Please try again.";
    errorMessageElement.style.display = "block";
    return;
  }

  // Step 8: Display recommendations
  if (recommendations.length === 0) {
    recommendationsElement.innerHTML = "<li>No valid recommendations found.</li>";
    return;
  }
  recommendations.forEach(book => {
    const li = document.createElement("li");
    const link = document.createElement("a");
    link.href = book.link;
    link.target = "_blank";
    link.textContent = `${book.title} by ${book.authors.join(", ")}`;
    li.appendChild(link);
    recommendationsElement.appendChild(li);
  });

  // Step 9: Send books for explanation (if applicable)
  try {
    let explanationData;
    if (tab === "lime") {
      explanationData = await fetchLimeExplanation('lime', recommendations);
      displayLimeExplanation(explanationData, 'lime');
    } else if (tab === "anchor") {
      explanationData = await fetchAnchorExplanation('anchor', recommendations, selectedBook.description);
      displayAnchorExplanation(explanationData);
    }
  } catch (error) {
    console.error("Error fetching explanation:", error);
    errorMessageElement.textContent = "Error fetching explanations. Please try again.";
    errorMessageElement.style.display = "block";
  }
}

async function fetchLimeExplanation(type, recommendations) {
  // Validate recommendations before proceeding
  if (!validateRecommendations(recommendations)) {
    console.error("Invalid recommendations structure.");
    return {error: "Invalid recommendations structure"};
  }

  const url = `/${type}-explanation`;

  const response = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({recommendations}) // Send recommendations as the body
  });

  if (!response.ok) {
    console.error(`Error: ${response.status} - ${response.statusText}`);
    return {error: `Failed to fetch explanation for ${type}`};
  }
  const explanationData = await response.json();
  console.log("Explanation data received:", explanationData); // Log explanation data
  return explanationData;
}

async function fetchAnchorExplanation(type, recommendations, originalDescription = "") {
  const url = `/${type}-explanation`;

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ recommendations, original_description: originalDescription }),
    });

    console.log("Raw response object:", response);

    if (!response.ok) {
      console.error(`Error: ${response.status} - ${response.statusText}`);
      return { error: `Failed to fetch explanation for ${type}` };
    }

    const explanationData = await response.json();
    console.log("Explanation data received:", explanationData);
    return explanationData;
  } catch (error) {
    console.error("Error during fetchAnchorExplanation:", error);
    return { error: "An error occurred while fetching Anchor explanations." };
  }
}
function displayAnchorExplanation(data) {
  const explanationContainer = document.getElementById('anchorExplanationList');
  explanationContainer.innerHTML = ''; // Clear previous explanations

  if (!Array.isArray(data) || data.length === 0) {
    console.error("No valid Anchor explanation data received:", data);
    explanationContainer.innerHTML = '<li>No Anchor explanations available. Please try refreshing the page and searching again.</li>';
    return;
  }

  data.forEach((explanation, index) => {
    console.log("Processing explanation:", explanation);

    const title = document.createElement('h3');
    title.textContent = explanation.title || `Recommendation ${index + 1}`;

    const anchorWords = document.createElement('p');
    anchorWords.textContent = `Anchor Words: ${explanation.anchor_words || 'None'}`;

    const precision = document.createElement('p');
    const precisionValue = explanation.precision;
    console.log("Precision value:", precisionValue);
    if (typeof precisionValue === "number") {
      precision.textContent = `Precision: ${precisionValue.toFixed(2)}`;
    } else {
      precision.textContent = "Precision: N/A";
      console.warn(`Invalid precision value for explanation ${index + 1}:`, precisionValue);
    }

    explanationContainer.appendChild(title);
    explanationContainer.appendChild(anchorWords);
    explanationContainer.appendChild(precision);
  });
}
function displayLimeExplanation(data) {
  console.log("Data passed to displayLimeExplanation:", data); // Log the initial data received

  const explanationContainer = document.getElementById('limeExplanationList');

  if (!explanationContainer) {
    console.error("limeExplanationList element not found in DOM");
    return;
  }

  explanationContainer.innerHTML = ''; // Clear previous explanations

  if (!Array.isArray(data)) {
    console.error("Invalid data format for explanation display. Expected array, received:", typeof data);
    explanationContainer.innerHTML = "<li>No explanation data available.</li>";
    return;
  }

  console.log(`Number of explanations to display: ${data.length}`);

  data.forEach((explanationData, idx) => {
    console.log(`Processing explanation #${idx + 1}:`, explanationData);

    const recommendationTitle = explanationData.title || `Recommendation ${idx + 1}`;
    const titleElement = document.createElement('h3');
    titleElement.textContent = recommendationTitle;

    const generalExplanation = document.createElement('p');
    generalExplanation.textContent = explanationData.general_explanation || "No general explanation provided.";

    const explanationList = document.createElement('ul');
    const explanationOutput = explanationData.explanation_output || [];

    if (Array.isArray(explanationOutput) && explanationOutput.length > 0) {
      explanationOutput.forEach(item => {
        const listItem = document.createElement('li');
        listItem.textContent = item; // Directly render the string
        explanationList.appendChild(listItem);
      });
    } else {
      const noExplanationItem = document.createElement('li');
      noExplanationItem.textContent = "No significant features found in the explanation.";
      explanationList.appendChild(noExplanationItem);
    }

    explanationContainer.appendChild(titleElement);
    explanationContainer.appendChild(generalExplanation);
    explanationContainer.appendChild(explanationList);
  });
}
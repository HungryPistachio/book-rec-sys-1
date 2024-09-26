console.log("Script loaded successfully!");

// Tab data configuration for dynamic generation
const tabData = [
  {
    id: "noXAI",
    title: "No XAI Explanation",
    checkboxId: "includeSameAuthorNoXAI",
    checkboxLabel: "Include recommendations by the same author",
    buttonLabel: "Get Recommendations",
  },
  {
    id: "lime",
    title: "LIME Explanation",
    checkboxId: "includeSameAuthorLime",
    checkboxLabel: "Include recommendations by the same author",
    buttonLabel: "Get Recommendations",
  },
  {
    id: "shap",
    title: "SHAP Explanation",
    checkboxId: "includeSameAuthorShap",
    checkboxLabel: "Include recommendations by the same author",
    buttonLabel: "Get Recommendations",
  },
  {
    id: "counterfactual",
    title: "Counterfactual Explanation",
    checkboxId: "includeSameAuthorCounterfactual",
    checkboxLabel: "Include recommendations by the same author",
    buttonLabel: "Get Recommendations",
  }
];

// Dynamically generate tab content based on the tabData array
const tabContentContainer = document.getElementById("tabContentContainer");
tabContentContainer.innerHTML = tabData.map(tab => `
  <div id="${tab.id}" class="tab-content" style="display:none;">
    <h2>${tab.title}</h2>
    <input type="text" id="bookTitle${tab.id}" placeholder="Enter a book title">
    <div class="toggle-label">
      <label>
        <input type="checkbox" id="${tab.checkboxId}" checked>
        ${tab.checkboxLabel}
      </label>
    </div>
    <button onclick="getRecommendations('${tab.id}')">${tab.buttonLabel}</button>
    <ul id="recommendations${tab.id}"></ul>
    <div id="${tab.id}Explanation" class="${tab.id.toLowerCase()}-explanation" style="display:none;">
      <h2>${tab.title} for <span id="explainedTitle${tab.id}"></span></h2>
      <ul id="${tab.id}ExplanationList"></ul>
    </div>
  </div>
`).join('');

// Simplified event listener with arrow function
const tabs = document.querySelectorAll(".tab");

tabs.forEach(tab => {
  tab.addEventListener("click", event => {
    const tabName = event.target.getAttribute("data-tab");
    openTab(tabName, event);
  });
});
// Add event listeners for recommendation buttons
  const recommendationButtons = document.querySelectorAll(".recommend-button");

  recommendationButtons.forEach(button => {
    button.addEventListener("click", event => {
      const tabId = event.target.getAttribute("data-tab");
      getRecommendations(tabId);
    });
  });

// Function to handle tab switching
  function openTab(tabName, evt) {
    const tabContents = document.getElementsByClassName("tab-content");

    // Hide all tab contents
    for (let i = 0; i < tabContents.length; i++) {
      tabContents[i].style.display = "none";
    }

    // Show the current tab content
    document.getElementById(tabName).style.display = "block";

    // Remove "active" class from all tabs
    const tabs = document.getElementsByClassName("tab");
    for (let i = 0; i < tabs.length; i++) {
      tabs[i].classList.remove("active");
    }

    // Add "active" class to the clicked tab
    evt.currentTarget.classList.add("active");
  }

// Function to clean and filter book descriptions
  function filterDescription(description) {
    const stopwords = ["the", "a", "an", "in", "on", "and", "or", "of", "to", "is", "for", "by"]; // Add more stopwords as needed
    return description
        .toLowerCase()
        .replace(/[^a-z\s]/g, '') // Remove non-alphabet characters
        .split(/\s+/) // Split by whitespace
        .filter(word => word.length > 0 && !stopwords.includes(word)) // Filter out stopwords and empty words
        .join(' '); // Rejoin the words
  }

// Vectorizes keywords for cosine similarity
  function vectorizeKeywords(keywords, allDocuments) {
    const documents = allDocuments.map(doc => doc.split(' '));
    const tfidfVectors = computeTFIDF(documents);

    const keywordsArray = keywords.split(' ');
    const vector = {};

    for (let i = 0; i < tfidfVectors.length; i++) {
      for (let word in tfidfVectors[i]) {
        vector[word] = tfidfVectors[i][word];
      }
    }

    // Handle case where the keywords result in an empty or single-word vector
    if (keywordsArray.length === 0) {
      console.error("Empty or invalid keyword vector");
      return new Array(Object.keys(vector).length).fill(0);  // Return a zero-filled vector of the same length
    }

    // Logging statements outside the loop
    console.log(`Vectorizing keywords: ${keywords}`);
    console.log(`All descriptions length: ${allDocuments.length}`);

    return keywordsArray.map(word => vector[word] || 0);
  }

// Computes term frequency (TF) for each word in a document
  function computeTF(words) {
    const tf = {};
    const totalWords = words.length;
    words.forEach(word => {
      tf[word] = (tf[word] || 0) + 1;
    });
    for (let word in tf) {
      tf[word] = tf[word] / totalWords;
    }
    return tf;
  }

// Computes inverse document frequency (IDF)
  function computeIDF(documents) {
    const idf = {};
    const totalDocs = documents.length;
    const wordDocCount = {};

    documents.forEach(doc => {
      const uniqueWords = new Set(doc);
      uniqueWords.forEach(word => {
        wordDocCount[word] = (wordDocCount[word] || 0) + 1;
      });
    });

    for (let word in wordDocCount) {
      idf[word] = Math.log(totalDocs / (1 + wordDocCount[word]));
    }
    return idf;
  }

// Computes TF-IDF vectors for a set of documents
  function computeTFIDF(documents) {
    const idf = computeIDF(documents);
    return documents.map(doc => {
      const tf = computeTF(doc);
      const tfidf = {};
      for (let word in tf) {
        tfidf[word] = tf[word] * idf[word];
      }
      return tfidf;
    });
  }

// Computes cosine similarity between two vectors
  function cosineSimilarity(vecA, vecB) {
    const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
    const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
    const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));

    // Logging before return
    console.log(`Cosine similarity calculation for vectors: ${vecA} and ${vecB}`);

    return dotProduct / (magnitudeA * magnitudeB);
  }

// Fetches recommendations from the Google Books API and processes XAI explanations
  async function getRecommendations(model) {
    const inputId = `bookTitle${model.charAt(0).toUpperCase() + model.slice(1)}`;
    const bookTitle = document.getElementById(inputId).value.toLowerCase();
    const recommendationsId = `recommendations${model.charAt(0).toUpperCase() + model.slice(1)}`;
    const recommendationsList = document.getElementById(recommendationsId);
    recommendationsList.innerHTML = '';  // Clear previous recommendations

    const includeSameAuthor = document.getElementById(`includeSameAuthor${model.charAt(0).toUpperCase() + model.slice(1)}`).checked;

    // Fetch book data from Google Books API
    const googleApiKey = 'AIzaSyDQTQy3LT7BOO_LLClbWuEvqiPbUbWWKBs';
    const googleApiUrl = `https://www.googleapis.com/books/v1/volumes?q=${encodeURIComponent(bookTitle)}&maxResults=40&key=${googleApiKey}`;

    let books;
    try {
      const apiResponse = await fetch(googleApiUrl);
      const apiData = await apiResponse.json();
      books = apiData.items || [];
    } catch (error) {
      console.error("Error fetching books from Google API", error);
      return;
    }

    const processedBooks = books.map(book => ({
      title: book.volumeInfo.title || "Unknown Title",
      authors: book.volumeInfo.authors || [],
      description: book.volumeInfo.description || "No description available",
      link: book.volumeInfo.infoLink || "#"
    }));

    // Find the selected book (first book that matches user input)
    const selectedBook = processedBooks.find(book => book.title.toLowerCase() === bookTitle);

    if (!selectedBook) {
      console.error("Selected book not found");
      return;
    }

    const allDescriptions = processedBooks.map(book => filterDescription(book.description));
    const keywordVector = vectorizeKeywords(selectedBook.description, allDescriptions);

    // Find recommendations based on cosine similarity
    const seenTitles = new Set();
    const recommendations = [];

// Ensure we're not recommending the same book
    processedBooks.forEach((book, idx) => {
      const bookVector = vectorizeKeywords(filterDescription(book.description), allDescriptions);
      const similarity = cosineSimilarity(keywordVector, bookVector);

      const normalizedTitle = book.title.toLowerCase();
      const isSameAuthor = book.authors.some(author => processedBooks[0].authors.includes(author));

      if (similarity > 0.5 && normalizedTitle !== bookTitle) { // Ensure it's not the same book
        if (!seenTitles.has(normalizedTitle)) {
          if (includeSameAuthor || !isSameAuthor) { // Respect the same author toggle
            seenTitles.add(normalizedTitle);
            recommendations.push(book);
          }
        }
      }
    });

// Handle empty recommendations
    if (recommendations.length === 0) {
      recommendationsList.innerHTML = "<li>No valid recommendations found.</li>";
      return;
    }

// Display recommendations
    recommendations.forEach(bookInfo => {
      const li = document.createElement("li");
      const link = document.createElement("a");
      link.href = bookInfo.link;
      link.target = "_blank";
      link.textContent = `${bookInfo.title} by ${bookInfo.authors.join(', ')}`;
      li.appendChild(link);
      recommendationsList.appendChild(li);
    });

  }

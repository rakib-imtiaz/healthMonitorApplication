var quotes = [
    "Quote 1",
    "Quote 2",
    "Quote 3",
    "Quote 4",
    "Quote 5"
];

var quoteIndex = 0;
var quoteElement = document.getElementById("quote");

function slideQuotes() {
    quoteElement.innerHTML = quotes[quoteIndex];
    quoteIndex = (quoteIndex + 1) % quotes.length;
}

setInterval(slideQuotes, 3000);
function changeLoadingColor() {
    var loadingDiv = document.getElementById("loading");
    var colors = ["red", "blue", "green", "yellow", "orange"];
    var currentIndex = 0;

    setInterval(function() {
        loadingDiv.style.backgroundColor = colors[currentIndex];
        currentIndex = (currentIndex + 1) % colors.length;
    }, 1000); // Change color every 1 second
}

// Function to show the result after a delay
function showResultWithDelay() {
    var loadingDiv = document.getElementById("loading");
    var resultDiv = document.getElementById("result");

    setTimeout(function() {
        loadingDiv.style.display = "none";
        resultDiv.style.display = "block";
    }, 3000); // Show result after 3 seconds
}

// Call the functions on page load
window.onload = function() {
    changeLoadingColor();
    showResultWithDelay();
};

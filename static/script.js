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

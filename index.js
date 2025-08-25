// const heading = document.createElement('h1');
// heading.textContent = 'Hello, World!';
// document.getElementById('root').appendChild(heading);

const button = document.getElementById('btn');
button.addEventListener('click', () => {
    alert('Button Clicked!');
});

const title = document.createElement("h2");
title.innerHTML = "This is a paragraph rendering within the div id container";

const heading = document.createElement('h1');
heading.textContent = 'Hello, World!';

root.append(title, heading);

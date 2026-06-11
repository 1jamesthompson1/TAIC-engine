(function () {
  function addHomeButton() {
    var header = document.querySelector(".md-header__inner");
    if (!header) {
      return;
    }

    if (document.querySelector(".docs-home-button")) {
      return;
    }

    var container = document.createElement("div");
    container.className = "docs-home-button";

    var appLink = document.createElement("a");
    appLink.className = "md-button md-button--primary";
    appLink.href = "https://ask.taic.org.nz";
    appLink.target = "_blank";
    appLink.rel = "noopener noreferrer";
    appLink.textContent = "Smart Tools App";
    container.appendChild(appLink);

    header.appendChild(container);
  }

  document.addEventListener("DOMContentLoaded", addHomeButton);
})();

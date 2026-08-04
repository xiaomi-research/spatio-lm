const header = document.querySelector("[data-header]");
const menuButton = document.querySelector(".menu-toggle");
const navigation = document.querySelector("#site-nav");
const copyButton = document.querySelector("[data-copy]");

const setMenu = (open) => {
  menuButton.setAttribute("aria-expanded", String(open));
  navigation.classList.toggle("is-open", open);
  document.body.classList.toggle("menu-open", open);
};

menuButton.addEventListener("click", () => {
  setMenu(menuButton.getAttribute("aria-expanded") !== "true");
});

navigation.querySelectorAll("a").forEach((link) => {
  link.addEventListener("click", () => setMenu(false));
});

window.addEventListener("resize", () => {
  if (window.innerWidth > 900) setMenu(false);
});

const updateHeader = () => header.classList.toggle("is-scrolled", window.scrollY > 12);
updateHeader();
window.addEventListener("scroll", updateHeader, { passive: true });

document.querySelectorAll("[data-tabs]").forEach((tabGroup) => {
  const tabs = [...tabGroup.querySelectorAll(":scope > .tab-list [role='tab'], :scope > .gallery-heading .tab-list [role='tab']")];
  const panels = [...tabGroup.querySelectorAll(":scope > [data-panel]")];

  const activateTab = (activeTab) => {
    tabs.forEach((tab) => {
      const isActive = tab === activeTab;
      tab.setAttribute("aria-selected", String(isActive));
      tab.tabIndex = isActive ? 0 : -1;
    });

    panels.forEach((panel) => {
      const isActive = panel.id === activeTab.dataset.tab;
      panel.classList.toggle("is-active", isActive);
      panel.setAttribute("aria-hidden", String(!isActive));
    });
  };

  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => activateTab(tab));
    tab.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      const direction = event.key === "ArrowRight" ? 1 : -1;
      const nextTab = tabs[(index + direction + tabs.length) % tabs.length];
      activateTab(nextTab);
      nextTab.focus();
    });
  });

  activateTab(tabs.find((tab) => tab.getAttribute("aria-selected") === "true") || tabs[0]);
});

const revealElements = document.querySelectorAll(".reveal");

if (window.location.hash) {
  revealElements.forEach((element) => element.classList.add("is-visible"));
} else if ("IntersectionObserver" in window) {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      });
    },
    { threshold: 0.08, rootMargin: "0px 0px -30px" },
  );

  revealElements.forEach((element) => observer.observe(element));
} else {
  revealElements.forEach((element) => element.classList.add("is-visible"));
}

copyButton.addEventListener("click", async () => {
  const citation = document.querySelector("#bibtex").textContent;

  try {
    await navigator.clipboard.writeText(citation);
    copyButton.textContent = "Copied";
    window.setTimeout(() => {
      copyButton.textContent = "Copy";
    }, 1800);
  } catch {
    copyButton.textContent = "Select text";
  }
});

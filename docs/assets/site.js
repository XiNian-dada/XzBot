(() => {
  const normalizePath = (pathname) => pathname.replace(/\/index\.html$/, "/");
  const currentPath = normalizePath(window.location.pathname);

  document.querySelectorAll("[data-nav]").forEach((node) => {
    const target = node.getAttribute("data-nav");
    if (target && currentPath.endsWith(target)) {
      node.classList.add("active");
    }
  });

  const sectionLinks = Array.from(document.querySelectorAll("[data-section]"));
  if (!sectionLinks.length) {
    return;
  }

  const sections = sectionLinks
    .map((link) => {
      const id = link.getAttribute("data-section");
      return id ? document.getElementById(id) : null;
    })
    .filter(Boolean);

  const setActiveSection = (id) => {
    sectionLinks.forEach((link) => {
      link.classList.toggle("active", link.getAttribute("data-section") === id);
    });
  };

  const fromHash = window.location.hash.slice(1);
  if (fromHash) {
    setActiveSection(fromHash);
  }

  if ("IntersectionObserver" in window && sections.length) {
    let activeId = fromHash || null;
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio);
        if (!visible.length) {
          return;
        }

        const id = visible[0].target.id;
        if (id && id !== activeId) {
          activeId = id;
          setActiveSection(id);
        }
      },
      {
        rootMargin: "-18% 0px -70% 0px",
        threshold: [0.1, 0.25, 0.5],
      },
    );

    sections.forEach((section) => observer.observe(section));
  }

  window.addEventListener("hashchange", () => {
    const id = window.location.hash.slice(1);
    if (id) {
      setActiveSection(id);
    }
  });
})();

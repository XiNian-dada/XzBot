(() => {
  const path = window.location.pathname.replace(/\/index\.html$/, '/');
  document.querySelectorAll('[data-nav]').forEach((node) => {
    const target = node.getAttribute('data-nav');
    if (target && path.endsWith(target)) {
      node.classList.add('active');
    }
  });
})();

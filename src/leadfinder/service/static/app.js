// The pages' one script (the CSP allows no other). Three things, none of which a page needs to work:
// a button that says it is busy after one click, the outline of dashes at the right edge, and sortable tables.
document.addEventListener('click', function (e) {
  var b = e.target.closest && e.target.closest('button[type=submit][data-busy]');
  if (b && b.form) {
    // one send per click: the form goes with this click, then the button is out of play
    setTimeout(function () { b.textContent = b.dataset.busy; b.disabled = true; }, 0);
  }
});

(function () {
  // the outline: a column of dashes at the right edge, one per section heading, that unfolds its labels when the
  // pointer comes near, with the section in view marked. Built from the headings, so a page with fewer than two has none.
  var main = document.querySelector('main');
  if (!main) return;
  var heads = Array.prototype.filter.call(main.querySelectorAll('h2.kicker[id]'), function (h) {
    return h.offsetParent !== null;
  });
  if (heads.length < 2) return;
  function label(h) {
    var c = h.cloneNode(true);
    Array.prototype.forEach.call(c.querySelectorAll('.info-wrap, .fn, sup'), function (x) { x.parentNode.removeChild(x); });
    var t = c.textContent.replace(/\s+/g, ' ').trim().split(' · ')[0];
    return t.charAt(0).toUpperCase() + t.slice(1);
  }
  var nav = document.createElement('nav');
  nav.className = 'outline';
  nav.setAttribute('aria-label', 'On this page');
  var ul = document.createElement('ul');
  var items = [{ id: 'top', text: 'Top', el: main }].concat(heads.map(function (h) { return { id: h.id, text: label(h), el: h }; }));
  var links = items.map(function (it) {
    var li = document.createElement('li'), a = document.createElement('a'), l = document.createElement('span'), d = document.createElement('span');
    a.href = '#' + it.id;
    l.className = 'outline__label'; l.textContent = it.text;
    d.className = 'outline__dash'; d.setAttribute('aria-hidden', 'true');
    a.appendChild(l); a.appendChild(d); li.appendChild(a); ul.appendChild(li);
    return a;
  });
  nav.appendChild(ul);
  document.body.appendChild(nav);
  if (!main.id) main.id = 'top';
  var current = -1;
  function mark() {
    var line = window.innerHeight * 0.35, at = 0;
    for (var i = 1; i < items.length; i++) {
      if (items[i].el.getBoundingClientRect().top <= line) at = i;
    }
    if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2) at = items.length - 1;
    if (at === current) return;
    current = at;
    links.forEach(function (a, i) {
      a.classList.toggle('is-active', i === at);
      if (i === at) a.setAttribute('aria-current', 'location'); else a.removeAttribute('aria-current');
    });
  }
  var queued = false;
  function onScroll() {
    if (queued) return;
    queued = true;
    requestAnimationFrame(function () { queued = false; mark(); });
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll);
  document.addEventListener('mousemove', function (e) {
    var r = nav.getBoundingClientRect();
    var near = e.clientX >= r.left - 120 && e.clientY >= r.top - 120 && e.clientY <= r.bottom + 120;
    nav.classList.toggle('is-near', near);
  }, { passive: true });
  mark();
})();

(function () {
  // a table marked data-sortable sorts by any column when its heading is clicked: numbers high to low first, words A to Z,
  // a second click reverses. A cell's data-v, when it has one, is what it sorts by; an empty one always goes last.
  Array.prototype.forEach.call(document.querySelectorAll('table[data-sortable]'), function (table) {
    var body = table.tBodies[0], heads = table.tHead ? table.tHead.rows[0].cells : [];
    if (!body) return;
    function value(row, i) {
      var cell = row.cells[i];
      if (!cell) return '';
      return (cell.hasAttribute('data-v') ? cell.getAttribute('data-v') : cell.textContent).trim();
    }
    function num(v) {
      var t = v.replace(/[^0-9.-]/g, '');
      return t === '' || t === '-' || t === '.' ? NaN : parseFloat(t);
    }
    Array.prototype.forEach.call(heads, function (th, i) {
      if (th.hasAttribute('data-nosort')) return;
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'sort';
      while (th.firstChild) b.appendChild(th.firstChild);
      th.appendChild(b);
      b.addEventListener('click', function () {
        var rows = Array.prototype.slice.call(body.rows);
        var numeric = rows.every(function (r) { var v = value(r, i); return v === '' || !isNaN(num(v)); });
        var was = th.getAttribute('aria-sort');
        var asc = was ? was === 'descending' : !numeric;
        rows.sort(function (x, y) {
          var a = value(x, i), c = value(y, i);
          if (a === '' || c === '') return a === c ? 0 : (a === '' ? 1 : -1);
          var d = numeric ? num(a) - num(c) : a.localeCompare(c);
          return asc ? d : -d;
        });
        rows.forEach(function (r) { body.appendChild(r); });
        Array.prototype.forEach.call(heads, function (h) { h.removeAttribute('aria-sort'); });
        th.setAttribute('aria-sort', asc ? 'ascending' : 'descending');
      });
    });
  });
})();

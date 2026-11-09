'use strict';
(async function () {
  const rows = document.getElementById('task-rows');
  const count = document.getElementById('results-count');
  const search = document.getElementById('task-search');
  const evidence = document.getElementById('evidence-filter');
  const sort = document.getElementById('task-sort');
  const more = document.getElementById('show-more');
  let expanded = false;
  let summary;
  const readable = name => name.replaceAll('_', ' ').replace(/^./, x => x.toUpperCase());
  function cell(text, className) {
    const node = document.createElement('td');
    node.textContent = text;
    if (className) node.className = className;
    return node;
  }
  function render() {
    const term = search.value.trim().toLowerCase().replaceAll('_', ' ');
    const selected = summary.tasks.filter(t => (readable(t.task).toLowerCase().includes(term) || String(t.id) === term)
      && (evidence.value === 'all' || (evidence.value === 'published' ? t.episodes > 0 : t.episodes === 0)));
    selected.sort((a, b) => sort.value === 'id' ? a.id - b.id : sort.value === 'name' ? a.task.localeCompare(b.task)
      : (b.mean_q ?? -1) - (a.mean_q ?? -1) || a.id - b.id);
    const shown = expanded ? selected : selected.slice(0, 12);
    rows.replaceChildren();
    for (const task of shown) {
      const row = document.createElement('tr');
      const title = cell('');
      const id = document.createElement('span');
      id.className = 'task-id';
      id.textContent = `TASK ${String(task.id).padStart(2, '0')}`;
      title.append(id, document.createTextNode(readable(task.task)));
      row.append(title, cell(task.release_route, 'route'), cell(task.episodes ? `${task.successes} / ${task.episodes}` : '—'));
      const score = cell('', 'score');
      if (task.mean_q !== null) {
        const value = document.createElement('span');
        value.className = 'score-number';
        value.textContent = task.mean_q.toFixed(3);
        const track = document.createElement('span');
        track.className = 'score-track';
        track.setAttribute('aria-hidden', 'true');
        const fill = document.createElement('i');
        fill.style.width = `${Math.max(0, Math.min(1, task.mean_q)) * 100}%`;
        track.append(fill);
        score.append(value, track);
      } else {
        score.textContent = '—';
      }
      const linkCell = cell('');
      if (task.episodes) {
        const link = document.createElement('a');
        link.href = `https://huggingface.co/datasets/${summary.dataset}/blob/${summary.revision}/${summary.folder}/json/${task.task}_301_0.json`;
        link.textContent = 'JSON ↗';
        link.setAttribute('aria-label', `${readable(task.task)} instance 301 JSON`);
        linkCell.append(link);
      } else {
        const missing = document.createElement('span');
        missing.className = 'missing';
        missing.textContent = 'Not published';
        linkCell.append(missing);
      }
      row.append(score, linkCell);
      rows.append(row);
    }
    count.textContent = `Showing ${shown.length} of ${selected.length} matching tasks · ${summary.published_tasks} tasks with public episodes`;
    more.hidden = expanded || shown.length === selected.length;
    if (!selected.length) {
      const row = document.createElement('tr');
      const empty = cell('No tasks match these filters.');
      empty.colSpan = 5;
      row.append(empty);
      rows.append(row);
    }
  }
  for (const control of [search, evidence, sort]) {
    control.addEventListener(control === search ? 'input' : 'change', () => { expanded = false; render(); });
  }
  more.addEventListener('click', () => { expanded = true; render(); });
  try {
    const response = await fetch('data/public-results.json');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    summary = await response.json();
    render();
  } catch (error) {
    count.textContent = 'The explorer could not load. Download the results JSON using the link below.';
    const row = document.createElement('tr');
    const linkCell = cell('');
    linkCell.colSpan = 5;
    const link = document.createElement('a');
    link.href = 'data/public-results.json';
    link.textContent = 'Download complete results data';
    linkCell.append(link); row.append(linkCell); rows.append(row);
  }
  document.querySelectorAll('[data-copy]').forEach(button => {
    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(document.getElementById(button.dataset.copy).textContent.trim());
        button.textContent = 'Copied';
        setTimeout(() => { button.textContent = 'Copy'; }, 2000);
      } catch (error) {
        button.textContent = 'Select code to copy';
      }
    });
  });
})();

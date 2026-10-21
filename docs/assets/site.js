  // Pilot styles (test seed 2001, starting battery 44%)
  const pilots = [["probe_hold", "Probe & hold"], ["steady_below", "Steady below cap"], ["shaky", "Shaky"], ["wavy", "Wavy"],
                  ["cruise", "Cruise"], ["blind_ppo", "PPO, memoryless"], ["replay_good", "Good PPO, replayed"],
                  ["switching", "Style switching"]];
  const pilotsEl = document.getElementById("pilots");
  if (pilotsEl) pilotsEl.innerHTML = pilots.map(([file, name]) =>
    `<div class="video-card"><video src="assets/videos/pilot_${file}.mp4" muted loop playsinline preload="metadata"></video>` +
    `<div class="label">${name}</div></div>`).join("");

  // Gallery: the same test flight (seed 2000) for each agent, by starting battery
  const agents = [["memoryless_ppo", "No battery information", "var(--red)", ""],
                  ["gru_labels_ppo", "Frozen GRU (ours)", "var(--accent)", ""],
                  ["true_battery_ppo", "True battery", "var(--purple)", "reference"],
                  ["ideal_ppo", "Ideal estimate", "var(--green)", "reference"],
                  ["gru_teacher_ppo", "Frozen GRU, ideal-estimate teacher", "#eb6834", "ablation"]];
  const gallery = document.getElementById("gallery");
  function showBattery(b) {
    if (!gallery) return;
    gallery.innerHTML = agents.map(([file, name, color, tag]) =>
      `<div class="video-card"><video src="assets/videos/${file}_b${b}.mp4" muted loop playsinline preload="metadata"></video>` +
      `<div class="label"><span class="dot" style="background:${color}"></span>${name}` +
      (tag ? `<span class="tag">${tag}</span>` : "") + `</div></div>`).join("");
    gallery.querySelectorAll("video").forEach(observe);
  }
  document.querySelectorAll(".chip[data-b]").forEach(chip => chip.addEventListener("click", () => {
    document.querySelectorAll(".chip[data-b]").forEach(c => c.setAttribute("aria-pressed", String(c === chip)));
    showBattery(chip.dataset.b);
  }));

  // Play videos only while they are on screen
  const watcher = "IntersectionObserver" in window ? new IntersectionObserver(entries => entries.forEach(e => {
    if (e.isIntersecting) { e.target.play().catch(() => {}); } else { e.target.pause(); }
  }), { threshold: 0.25 }) : null;
  function observe(video) { if (watcher) { watcher.observe(video); } else { video.autoplay = true; } }
  document.querySelectorAll("video").forEach(observe);
  showBattery("03");

  // Copy the BibTeX
  const copyButton = document.getElementById("copy");
  if (copyButton) copyButton.addEventListener("click", async () => {
    const button = document.getElementById("copy");
    try { await navigator.clipboard.writeText(document.getElementById("bibtex").textContent); button.textContent = "Copied"; }
    catch (err) { button.textContent = "Select and copy"; }
    setTimeout(() => { button.textContent = "Copy"; }, 1600);
  });

// Common JS shared by index.html and collective.html
(function () {
    const modelNameToKey = {
        'BlipForConditionalGeneration': 'blip',
        'Blip2ForConditionalGeneration': 'blip2',
        'Gemma3ForConditionalGeneration': 'gemma',
        'InternVLForConditionalGeneration': 'intern_vlm'
    };

    function setInitialModelSelection(modelSelectEl, currentModelEl) {
        if (!modelSelectEl || !currentModelEl) return;
        const initialKey = modelNameToKey[(currentModelEl.textContent || '').trim()];
        if (initialKey) modelSelectEl.value = initialKey;
    }

    function attachResetRedis(resetBtnEl, statusEl) {
        if (!resetBtnEl || !statusEl) return;
        resetBtnEl.addEventListener('click', async () => {
            statusEl.textContent = 'Resetting Redis cache...';
            try {
                const resp = await fetch('/reset-redis', {method: 'GET'});
                if (!resp.ok) throw new Error('Failed to reset Redis cache');
                const data = await resp.json();
                statusEl.textContent = data.message || 'Redis cache reset.';
            } catch (err) {
                statusEl.innerHTML = `<span class="error">${err.message}</span>`;
            }
        });
    }

    function attachModelSwitcher(modelSelectEl, currentModelEl, statusEl) {
        if (!modelSelectEl || !currentModelEl || !statusEl) return;
        modelSelectEl.addEventListener('change', async () => {
            const selectedModel = modelSelectEl.value;
            statusEl.textContent = 'Switching model...';
            try {
                const resp = await fetch('/set-model', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({model: selectedModel})
                });
                if (!resp.ok) throw new Error('Failed to switch model');
                const data = await resp.json();
                currentModelEl.textContent = data.model_name || selectedModel;
                // Update the current flag prompt if provided by the server
                const flagEl = document.getElementById('current-flag-prompt');
                if (flagEl) flagEl.textContent = data.flag_caption_prompt || '';
                // Also update the textarea value so users can see/edit it immediately
                const flagInput = document.getElementById('flag-prompt');
                if (flagInput) flagInput.value = data.flag_caption_prompt || '';
                statusEl.textContent = 'Model switched.';
            } catch (err) {
                statusEl.innerHTML = `<span class="error">${err.message}</span>`;
            }
        });
    }

    function attachSavePrompt(saveBtnEl, inputEl, currentPromptEl, promptStatusEl) {
        if (!saveBtnEl || !inputEl || !currentPromptEl || !promptStatusEl) return;
        saveBtnEl.addEventListener('click', async () => {
            const promptVal = inputEl.value;
            promptStatusEl.textContent = 'Saving...';
            try {
                const resp = await fetch('/set-caption-prompt', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({caption_prompt: promptVal})
                });
                if (!resp.ok) throw new Error('Failed to save caption prompt');
                const data = await resp.json();
                currentPromptEl.textContent = data.caption_prompt || '';
                promptStatusEl.textContent = 'Saved';
            } catch (err) {
                promptStatusEl.innerHTML = `<span class='error'>${err.message}</span>`;
            }
        });
    }

    function attachSaveFlagPrompt(saveBtnEl, inputEl, currentFlagPromptEl, promptStatusEl) {
        if (!saveBtnEl || !inputEl || !currentFlagPromptEl || !promptStatusEl) return;
        saveBtnEl.addEventListener('click', async () => {
            const promptVal = inputEl.value;
            promptStatusEl.textContent = 'Saving...';
            try {
                const resp = await fetch('/set-flag-prompt', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({flag_caption_prompt: promptVal})
                });
                if (!resp.ok) throw new Error('Failed to save flag prompt');
                const data = await resp.json();
                currentFlagPromptEl.textContent = data.flag_caption_prompt || '';
                // Reflect normalized value back into the textarea
                inputEl.value = data.flag_caption_prompt || '';
                promptStatusEl.textContent = 'Saved';
            } catch (err) {
                promptStatusEl.innerHTML = `<span class='error'>${err.message}</span>`;
            }
        });
    }

    window.CommonUI = {
        modelNameToKey,
        setInitialModelSelection,
        attachResetRedis,
        attachModelSwitcher,
        attachSavePrompt,
        attachSaveFlagPrompt,
    };
})();

// Page-specific JS for index.html (migrated from inline script)
(function () {
    document.addEventListener('DOMContentLoaded', function () {
        const form = document.getElementById('upload-form');
        const fileInput = document.getElementById('file-input');
        const statusEl = document.getElementById('status');
        const resultsEl = document.getElementById('results');
        const previewsEl = document.getElementById('previews');
        const submitBtn = document.getElementById('submit-btn');
        const modelSelect = document.getElementById('model-select');
        const currentModelEl = document.getElementById('current-model');
        const captionPromptInput = document.getElementById('caption-prompt');
        const savePromptBtn = document.getElementById('save-prompt-btn');
        const currentPromptEl = document.getElementById('current-prompt');
        const promptStatusEl = document.getElementById('prompt-status');
        const flagPromptInput = document.getElementById('flag-prompt');
        const saveFlagPromptBtn = document.getElementById('save-flag-prompt-btn');
        const currentFlagPromptEl = document.getElementById('current-flag-prompt');
        const flagPromptStatusEl = document.getElementById('flag-prompt-status');
        const modeToggle = document.getElementById('mode-toggle');
        const modeWarning = document.getElementById('mode-warning');

        const resetRedisBtn = document.getElementById('reset-redis-btn');
        const {setInitialModelSelection, attachResetRedis, attachModelSwitcher} = window.CommonUI || {};
        if (window.CommonUI) {
            setInitialModelSelection(modelSelect, currentModelEl);
            attachResetRedis(resetRedisBtn, statusEl);
            attachModelSwitcher(modelSelect, currentModelEl, statusEl);
        }

        const {attachSavePrompt, attachSaveFlagPrompt, modelNameToKey} = window.CommonUI || {};
        if (window.CommonUI) {
            attachSavePrompt(savePromptBtn, captionPromptInput, currentPromptEl, promptStatusEl);
            attachSaveFlagPrompt(saveFlagPromptBtn, flagPromptInput, currentFlagPromptEl, flagPromptStatusEl);
        }

        function isCollectiveSupported() {
            const modelName = (currentModelEl.textContent || '').trim();
            const key = (modelNameToKey && modelNameToKey[modelName]) || null;
            return key === 'gemma' || key === 'intern_vlm';
        }

        const collectiveSupportedKeys = ['gemma', 'intern_vlm'];

        function filterModelOptionsForMode(on) {
            if (!modelSelect) return;
            const options = Array.from(modelSelect.options || []);
            options.forEach(opt => {
                const supported = collectiveSupportedKeys.includes(opt.value);
                const hide = on && !supported;
                opt.hidden = hide;
                opt.disabled = hide;
            });
            // Auto-switch to a supported model when entering collective mode
            if (on && !collectiveSupportedKeys.includes(modelSelect.value)) {
                const fallback = options.find(o => collectiveSupportedKeys.includes(o.value));
                if (fallback && modelSelect.value !== fallback.value) {
                    modelSelect.value = fallback.value;
                    modelSelect.dispatchEvent(new Event('change', {bubbles: true}));
                }
            }
        }

        function updateModeUI() {
            const on = !!modeToggle.checked;
            // Filter model options based on mode
            filterModelOptionsForMode(on);

            document.body.classList.toggle('collective-on', on);
            resultsEl.className = on ? 'result' : '';
            if (on && !isCollectiveSupported()) {
                modeWarning.style.display = 'inline';
                submitBtn.disabled = true;
            } else {
                modeWarning.style.display = 'none';
                submitBtn.disabled = false;
            }
            submitBtn.textContent = on ? 'Generate Collective Caption' : 'Caption Images';
        }

        // Update UI when toggle changes
        modeToggle && modeToggle.addEventListener('change', updateModeUI);
        // Observe model name changes to re-evaluate support
        if (currentModelEl) {
            const mo = new MutationObserver(() => updateModeUI());
            mo.observe(currentModelEl, {childList: true, subtree: true, characterData: true});
        }
        // Initialize
        updateModeUI();

        function resetUI() {
            statusEl.textContent = '';
            resultsEl.innerHTML = '';
            resultsEl.style.display = 'none';
            previewsEl.innerHTML = '';
        }

        fileInput && fileInput.addEventListener('change', () => {
            previewsEl.innerHTML = '';
            const files = Array.from(fileInput.files || []);
            files.forEach((file, i) => {
                const url = URL.createObjectURL(file);
                const card = document.createElement('div');
                card.className = 'card';
                card.dataset.index = String(i);
                card.innerHTML = `
                        <img class="preview" src="${url}" alt="preview" />
                        <div style="flex:1">
                            <div><strong>${file.name}</strong></div>
                            <div class="small" data-role="flag"></div>
                            <div class="small">${(file.size / 1024).toFixed(1)} KB</div>
                            <div class="small" data-role="caption">Caption: (pending...)</div>
                            <div class="tags" data-role="tags"></div>
                        </div>
                    `;
                previewsEl.appendChild(card);
            });
            // Clear any previous collective result
            resultsEl.innerHTML = '';
            resultsEl.style.display = 'none';
        });

        form && form.addEventListener('submit', async (e) => {
            e.preventDefault();
            resultsEl.innerHTML = '';

            const files = fileInput.files;
            if (!files || files.length === 0) {
                statusEl.innerHTML = '<span class="error">Please choose at least one image.</span>';
                return;
            }

            const formData = new FormData();
            for (const f of files) formData.append('images', f);

            submitBtn.disabled = true;
            const collective = !!modeToggle.checked;
            statusEl.textContent = collective ? 'Uploading and generating collective caption...' : 'Uploading and captioning...';

            try {
                if (collective) {
                    const resp = await fetch('/caption-collective-images', {method: 'POST', body: formData});
                    const text = await resp.text();
                    if (!resp.ok) throw new Error(`Server error (${resp.status}): ${text}`);
                    const data = JSON.parse(text);
                    const caption = data.collective_caption || '';
                    const count = data.count || (files ? files.length : 0);
                    const tags = Array.isArray(data.tags) ? data.tags : [];
                    const tagsHtml = tags.length ? `<div class="tags" style="margin-top:6px;">${tags.map(t => `<span class="tag">${t}</span>`).join('')}</div>` : '';
                    if (data.flagged === true) {
                        resultsEl.innerHTML = `<div style="margin-top:10px; color:green; font-weight:bold;">Flag: True</div><br/>`;
                    } else if (data.flagged === false) {
                        resultsEl.innerHTML = `<div style="margin-top:10px; color:red; font-weight:bold;">Flag: False</div><br/>`;
                    }
                    resultsEl.innerHTML += `<div><strong>Collective Caption</strong> (${count} image${count === 1 ? '' : 's'}):</div><div style="margin-top:6px;">${caption}</div>${tagsHtml}`;
                    resultsEl.style.display = 'block';
                    statusEl.textContent = 'Done';
                } else {
                    const resp = await fetch('/caption-images', {method: 'POST', body: formData});
                    if (!resp.ok) {
                        const t = await resp.text();
                        throw new Error(`Server error (${resp.status}): ${t}`);
                    }
                    const data = await resp.json();
                    statusEl.textContent = 'Done';
                    (data.results || []).forEach((item, i) => {
                        const card = previewsEl.children[i];
                        if (!card) return;
                        const captionEl = card.querySelector('[data-role="caption"]');
                        const tagsEl = card.querySelector('[data-role="tags"]');
                        const flagEl = card.querySelector('[data-role="flag"]');
                        if (captionEl) captionEl.textContent = `Caption: ${item.caption || ''}`;
                        if (tagsEl) {
                            tagsEl.innerHTML = (item.tags || []).map(t => `<span class="tag">${t}</span>`).join('');
                        }
                        if (flagEl) {
                            if (item.flagged === true) {
                                flagEl.innerHTML = '<span style="color:#228B22;font-weight:bold;">Flag: True</span>';
                            } else if (item.flagged === false) {
                                flagEl.innerHTML = '<span style="color:#b00020;font-weight:bold;">Flag: False</span>';
                            } else {
                                flagEl.innerHTML = '';
                            }
                        }
                    });
                }
            } catch (err) {
                console.error(err);
                statusEl.innerHTML = `<span class=\"error\">${err.message}</span>`;
            } finally {
                submitBtn.disabled = false;
            }
        });
    });
})();

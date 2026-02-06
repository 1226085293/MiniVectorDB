#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

// -------- 参数解析 --------
const args = process.argv.slice(2);

function getArgValues(flag) {
    const index = args.indexOf(flag);
    if (index === -1) return [];
    const values = [];
    for (let i = index + 1; i < args.length; i++) {
        if (args[i].startsWith('-')) break;
        values.push(args[i]);
    }
    return values;
}

const extensions = getArgValues('-e').map(e => e.replace('.', ''));
const targets = getArgValues('-d');

if (!extensions.length || !targets.length) {
    console.error('用法: node mdg -e ts js -d ./a ./b ./file.ts');
    process.exit(1);
}

// -------- 文件收集 --------
const collectedFiles = [];

function collect(targetPath) {
    if (!fs.existsSync(targetPath)) return;

    const stat = fs.statSync(targetPath);

    if (stat.isFile()) {
        const ext = path.extname(targetPath).slice(1);
        if (extensions.includes(ext)) {
            collectedFiles.push(targetPath);
        }
        return;
    }

    if (stat.isDirectory()) {
        const items = fs.readdirSync(targetPath);
        items.forEach(item => {
            collect(path.join(targetPath, item));
        });
    }
}

targets.forEach(t => collect(t));

// -------- 生成 Markdown --------
let mdContent = '';

collectedFiles.forEach((filePath, index) => {
    const content = fs.readFileSync(filePath, 'utf-8');
    const relativePath = path.relative(process.cwd(), filePath);

    mdContent += `\`\`\`${relativePath}
${content}
\`\`\`
`;

    if (index !== collectedFiles.length - 1) {
        mdContent += `\n---\n\n`;
    }
});

// -------- 输出 --------
const now = new Date();

const pad = (n) => String(n).padStart(2, '0');

const timestamp =
    `${now.getFullYear()}-` +
    `${pad(now.getMonth() + 1)}-` +
    `${pad(now.getDate())}-` +
    `${pad(now.getHours())}-` +
    `${pad(now.getMinutes())}-` +
    `${pad(now.getSeconds())}`;

const outputFile = `merged_${timestamp}.md`;

fs.writeFileSync(path.join("./temp", outputFile), mdContent, 'utf-8');

console.log(`✅ 已生成 Markdown 文件: ${outputFile}`);
console.log(`📄 共处理 ${collectedFiles.length} 个文件`);
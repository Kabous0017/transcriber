# samples/

Drop short audio files here for quick pipeline tests. Contents are gitignored by default so
private recordings never end up in a public repo.

To include a sample in the repo (e.g. a clearly-licensed demo clip), add an explicit exception in
`.gitignore`:

```
!samples/my-creative-commons-clip.mp3
```

## Usage

```bash
transcriber run samples/your-file.mp3 --min-speakers 2 --max-speakers 6
```

Output lands in `outputs/your-file.{json,txt,srt}`.

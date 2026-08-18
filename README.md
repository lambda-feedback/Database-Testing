# Database-Testing
Test evaluation functions against previous student submissions.

## muEd mode

`--api_mode mued` tests the [muEd](https://mued.org) `EvaluateRequest`/`Feedback[]` contract instead of the legacy Lambda-Feedback one. Payloads are validated against the live [`mued-api/spec`](https://github.com/mued-api/spec) OpenAPI schema, vendored here as a git submodule at `vendor/mued-api`, so schema drift fails loudly instead of silently.

One-time local setup (CI does this automatically):

```
git submodule update --init
cd vendor/mued-api && npm ci && npm run bundle
```

Re-run the bundle step whenever the submodule pointer is bumped. To pull in schema updates:

```
cd vendor/mued-api && git checkout main && git pull
cd ../.. && git add vendor/mued-api && git commit -m "Bump mued-api/spec submodule"
```

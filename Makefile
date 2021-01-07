PHONY: run run-container gcloud-deploy

run:
	@streamlit run streamlit_app.py --server.port=8080 --server.address=0.0.0.0

run-container:
	@docker build . -t mudes_ui
	@docker run -p 8080:8080 mudes_ui

gcloud-deploy:
	@gcloud app deploy app.yaml
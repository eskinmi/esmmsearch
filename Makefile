.PHONY: benchmark-setup benchmark-train benchmark-clean

# Download and preprocess Expedia dataset
benchmark-setup:
	@echo "Downloading Expedia Personalized Sort dataset..."
	cd benchmark && kaggle competitions download -c expedia-personalized-sort
	@echo "Extracting data..."
	cd benchmark && unzip -o expedia-personalized-sort.zip -d dataset/
	cd benchmark/dataset && unzip -o data.zip && rm -f *.zip
	@echo "Preprocessing..."
	uv run python benchmark/bin/preprocess.py --input benchmark/dataset --output benchmark/dataset/processed
	@echo "Done! Data ready at benchmark/dataset/processed/"

# Train on Expedia benchmark
benchmark-train:
	@UUID=$$(python3 -c 'import uuid; print(uuid.uuid4())'); \
	echo "Run UUID: $$UUID"; \
	echo "To view TensorBoard for this run, run:"; \
	echo "  tensorboard --logdir runs/$$UUID"; \
	ESMM_CTR_LOSS_WEIGHT=1.0; \
	ESMM_CTCVR_LOSS_WEIGHT=5.0; \
	uv run esmmrank --data expedia --data-dir benchmark/dataset/processed \
		--epochs 10 --lr 0.0005 --batch-size 512 \
		--log-path benchmark/log_$$UUID.json \
		--tensorboard-dir runs/$$UUID/ \
		--save-path benchmark/ckpt/best_model_$$UUID.pt

# Clean benchmark data
benchmark-clean:
	rm -rf benchmark/dataset/
	rm -f benchmark/expedia-personalized-sort.zip
	@echo "Benchmark data cleaned"

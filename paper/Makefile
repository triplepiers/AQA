.DEFAULT_GOALS := build

.PHONY:
clear:
	@rm -f *.aux *.bcf *.log *.xml *.toc

build:
	@rm -f zjuthesis.pdf
	@xelatex zjuthesis.tex
	@make clear

clean:
	@make clear
	@rm -f zjuthesis.pdf

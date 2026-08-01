// ============================================================================
//  Source.cpp - demonstration program for the module
//               "Shape detection of signs to support the segmentation"
//  Component : Member 4 shape/segmentation-support module
//  Subject   : UCCC2513 Mini Project - Traffic sign detection
//
//  The program reads every .png file found under a given directory
//  (default "Test_84_Signs", which holds the 84 test images in the three
//  sub-directories "Red Signs", "Blue Signs" and "Yellow Signs") and, for
//  each picture, shows the six stages of the pipeline:
//
//      1 Original                    4 Selected sign (shape fitted)
//      2 Colour candidate mask       5 Shape-constrained graph-cut mask
//      3 Colour + edge shape models  6 Boundary-safe segmented sign
//
//  Usage:
//      ShapeDetection.exe                       - interactive, default folder
//      ShapeDetection.exe "Test_84_Signs"       - interactive, given folder
//      ShapeDetection.exe "Test_84_Signs" -batch- no windows, writes Outputs\
//      ShapeDetection.exe --synthetic-test       - generate/evaluate exact masks
//      ShapeDetection.exe --evaluate-segmentation <root>
//      ShapeDetection.exe --evaluate-camvid <dataset-root> [--external-output <root>]
//      ShapeDetection.exe --self-test             - fast deterministic checks
//
//  Keys during the demo:  ESC = quit,  s = save current result,  any = next
// ============================================================================

#include	<opencv2/opencv.hpp>
#include	<opencv2/highgui/highgui.hpp>
#include	<opencv2/core/utils/filesystem.hpp>
#include	<iostream>
#include	<iomanip>
#include	<fstream>
#include	<map>
#include	<string>
#include	<vector>
#include	"Supp.h"
#include	"ShapeDetect.h"
#include	"SegmentationTest.h"

using namespace cv;
using namespace std;

// Return only the file name (no directory) of a full path.
static string baseName(const string &path) {
	size_t	pos = path.find_last_of("/\\");
	return (pos == string::npos) ? path : path.substr(pos + 1);
}

// Read the optional ground-truth file "<file name> <SHAPE>" used to score the
// module.  Missing file simply means "no scoring".
static map<string, SignShape> loadLabels(const string &path) {
	map<string, SignShape>	labels;
	ifstream				in(path.c_str());
	string					name, shape;

	if (!in.is_open()) return labels;
	while (in >> name) {
		if (name.empty() || name[0] == '#') {	// skip a comment line
			getline(in, shape);
			continue;
		}
		if (!(in >> shape)) break;
		if      (shape == "CIRCLE")    labels[name] = SHAPE_CIRCLE;
		else if (shape == "TRIANGLE")  labels[name] = SHAPE_TRIANGLE;
		else if (shape == "RECTANGLE") labels[name] = SHAPE_RECTANGLE;
		else if (shape == "OCTAGON")   labels[name] = SHAPE_OCTAGON;
	}
	return labels;
}

// Return the name of the directory holding the file, e.g. "Red Signs".
static string parentName(const string &path) {
	size_t	pos = path.find_last_of("/\\");
	if (pos == string::npos) return string(".");
	string	dir = path.substr(0, pos);
	size_t	pos2 = dir.find_last_of("/\\");
	return (pos2 == string::npos) ? dir : dir.substr(pos2 + 1);
}

int main(int argc, char** argv) {
	string			rootDir = "Test_84_Signs";
	bool			batch   = false;
	bool			verbose = false;
	bool			syntheticTest = false;
	bool			selfTest = false;
	string			evaluationDir;
	string			camVidDir;
	string			externalOutputDir = "External_Test_Results/CamVid";
	string			camVidSplit = "all";
	string			camVidLabelDir;
	double			camVidContext = 3.0;
	bool			saveExternalPredictions = true;
	bool			camVidBoxPrompt = false;

	for (int i = 1; i < argc; i++) {
		string	a = argv[i];
		if (a == "-batch" || a == "--batch") batch = true;
		else if (a == "-v" || a == "--verbose") verbose = true;
		else if (a == "--synthetic-test") syntheticTest = true;
		else if (a == "--self-test") selfTest = true;
		else if (a == "--evaluate-segmentation" && i + 1 < argc)
			evaluationDir = argv[++i];
		else if (a == "--evaluate-camvid" && i + 1 < argc)
			camVidDir = argv[++i];
		else if (a == "--external-output" && i + 1 < argc)
			externalOutputDir = argv[++i];
		else if (a == "--camvid-split" && i + 1 < argc)
			camVidSplit = argv[++i];
		else if (a == "--camvid-context" && i + 1 < argc)
			camVidContext = atof(argv[++i]);
		else if (a == "--camvid-label-root" && i + 1 < argc)
			camVidLabelDir = argv[++i];
		else if (a == "--no-external-predictions")
			saveExternalPredictions = false;
		else if (a == "--camvid-box-prompt")
			camVidBoxPrompt = true;
		else rootDir = a;
	}

	ShapeParams		param;					// all tunable values live here
	if (selfTest) return runShapeDetectionSelfTests();
	if (syntheticTest) {
		const string testRoot = "Synthetic_Background_Test";
		if (!generateSyntheticBackgroundDataset(testRoot, 64)) {
			cerr << "Could not generate the synthetic background test set.\n";
			return -2;
		}
		return evaluateSegmentationDataset(testRoot, param, true);
	}
	if (!evaluationDir.empty())
		return evaluateSegmentationDataset(evaluationDir, param, true);
	if (!camVidDir.empty())
		return evaluateCamVidDataset(camVidDir, externalOutputDir, param,
			saveExternalPredictions, camVidSplit, camVidContext, camVidLabelDir,
			camVidBoxPrompt);

	// ------------------------------------------------------------------
	// Collect the names of the input image files from the given directory
	// (the demo requirement of the assignment).
	// ------------------------------------------------------------------
	vector<string>	imageNames;
	try {											// glob throws on a bad folder
		glob(rootDir + "/*.png", imageNames, true);	// true = search sub-folders
		if (imageNames.empty())
			glob(rootDir + "/*.jpg", imageNames, true);
	}
	catch (const cv::Exception &) {
		imageNames.clear();
	}

	if (imageNames.empty()) {
		cout << "No image found under \"" << rootDir << "\".\n"
			 << "Run the program from the folder that contains \"" << rootDir
			 << "\", or pass the folder name as the first argument.\n";
		return -1;
	}
	cout << imageNames.size() << " image(s) found under \"" << rootDir << "\"\n\n";

	const string	outDir = "Outputs";
	if (batch) utils::fs::createDirectories(outDir);
	ofstream refinementCsv;
	if (batch) {
		refinementCsv.open((outDir + "/refinement_diagnostics.csv").c_str());
		refinementCsv << "file,shape,source,prior_area,graphcut_area,final_area,"
			"area_ratio,core_recall,color_recall,span_ratio,prior_scale,accepted,decision\n";
	}

	int				shapeCount[5] = { 0, 0, 0, 0, 0 };
	int				detected = 0;
	int				nonEmptyMasks = 0, binaryMasks = 0, oneComponentMasks = 0;
	int				blackOutsideMasks = 0, sourceIdenticalMasks = 0;

	// optional scoring against the hand-labelled shapes
	map<string, SignShape>	labels = loadLabels("shape_labels.txt");
	int						confusion[5][5] = { { 0 } };
	int						scored = 0, correct = 0;
	if (!labels.empty())
		cout << labels.size() << " ground-truth labels read from "
			 << "\"shape_labels.txt\"\n\n";

	for (size_t i = 0; i < imageNames.size(); i++) {
		Mat		srcI = imread(imageNames[i]);
		if (srcI.empty()) {
			cout << "cannot open " << imageNames[i] << endl;
			continue;
		}

		cout << "[" << i + 1 << "/" << imageNames.size() << "] "
			 << parentName(imageNames[i]) << " / " << baseName(imageNames[i])
			 << "  (" << srcI.cols << "x" << srcI.rows << ")\n";

		// ------------------------------------------------------------------
		// Two display windows, in the style of the lecture examples.
		// ------------------------------------------------------------------
		int const	noOfImagePerCol = 2, noOfImagePerRow = 3;
		Mat			detailWin, win[noOfImagePerRow * noOfImagePerCol],
					legend[noOfImagePerRow * noOfImagePerCol];
		createWindowPartition(srcI, detailWin, win, legend,
			noOfImagePerCol, noOfImagePerRow);

		putText(legend[0], "1 Original", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
		putText(legend[1], "2 Colour candidate mask", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
		putText(legend[2], "3 Fused colour + edge shapes", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
		putText(legend[3], "4 Selected sign", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
		putText(legend[4], "5 Graph-cut refined mask", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
		putText(legend[5], "6 Pixel-accurate segmentation", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

		int const	noOfImagePerCol2 = 1, noOfImagePerRow2 = 2;
		Mat			resultWin, win2[noOfImagePerRow2 * noOfImagePerCol2],
					legend2[noOfImagePerRow2 * noOfImagePerCol2];
		createWindowPartition(srcI, resultWin, win2, legend2,
			noOfImagePerCol2, noOfImagePerRow2);
		putText(legend2[0], "Original", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
		putText(legend2[1], "Sign segmented by shape", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

		srcI.copyTo(win[0]);
		srcI.copyTo(win2[0]);

		// ------------------------------------------------------------------
		// Stage 1 - colour candidate masks (support stage), one per colour
		// ------------------------------------------------------------------
		vector<Mat>	masks(3);
		buildColorMasks(srcI, param, masks[0], masks[1], masks[2]);

		Mat		colorMask = masks[0] | masks[1] | masks[2];	// for display only
		Mat		show;
		cvtColor(colorMask, show, COLOR_GRAY2BGR);
		show.copyTo(win[1]);

		// ------------------------------------------------------------------
		// Stage 2 - boundary following + shape classification, per colour so
		// that touching regions of different colours never merge
		// ------------------------------------------------------------------
		vector<ShapeInfo>	dropped;
		Mat				edgeMap;
		vector<ShapeInfo>	shapes = detectSignShapesAdvanced(
			srcI, masks, param, &edgeMap, &dropped);

		Mat		canvas = srcI.clone();
		const size_t displayLimit = verbose ? shapes.size() : 3;
		for (size_t k = 0; k < shapes.size() && k < displayLimit; k++) {
			drawShape(canvas, shapes[k]);
			printShapeReport(shapes[k], (int)k);
		}
		if (shapes.size() > displayLimit)
			cout << "    ... " << (shapes.size() - displayLimit)
				 << " lower-ranked proposal(s) suppressed from the demo panel\n";
		if (shapes.empty()) {
			cout << "    no candidate region survived the size / convexity test";
			cout << (dropped.empty() ? " (the colour mask was empty)\n" : ":\n");
			for (size_t k = 0; k < dropped.size(); k++)
				printRejectReport(dropped[k], (int)k);
		}
		canvas.copyTo(win[2]);

		// ------------------------------------------------------------------
		// Stage 3 - choose the sign, Stage 4 - rebuild the mask, then segment
		// ------------------------------------------------------------------
		int			best  = selectBestSign(shapes, param);
		SignShape	found = SHAPE_UNKNOWN;
		Mat			finalMask = Mat::zeros(srcI.size(), CV_8U);
		Mat			finalSegmented = Mat::zeros(srcI.size(), srcI.type());
		RefinementDiagnostics refinement;

		if (best >= 0) {
			const ShapeInfo	&sign = shapes[best];

			Mat		pick = srcI.clone();
			drawShape(pick, sign, true, 2);
			pick.copyTo(win[3]);

			Mat		shapeMask = refineShapeMask(srcI, sign, masks, param, NULL,
				&refinement);
			shapeMask.copyTo(finalMask);
			cvtColor(shapeMask, show, COLOR_GRAY2BGR);
			show.copyTo(win[4]);

			Mat		segmented;
			srcI.copyTo(segmented, shapeMask);		// mask AND original image
			segmented.copyTo(finalSegmented);
			segmented.copyTo(win[5]);
			segmented.copyTo(win2[1]);

			found = sign.shape;
			shapeCount[sign.shape]++;
			if (sign.shape != SHAPE_UNKNOWN) detected++;

			cout << "    ==> selected region " << best << " : "
					 << shapeName(sign.shape) << "  (score " << fixed
					 << setprecision(2) << sign.score << ")\n";
			cout << "        refinement: " << refinement.decision
				 << " | prior scale " << setprecision(2) << refinement.priorScale
				 << " | area " << refinement.finalArea << "/" << refinement.priorArea
				 << " | core " << refinement.coreRecall
				 << " | colour " << refinement.colorRecall
				 << " | span " << refinement.spanRatio << "\n";
			if (refinementCsv.is_open()) {
				const char sourceMark = sign.proposalSource == SOURCE_EDGE ? 'E' :
					(sign.proposalSource == SOURCE_HOUGH ? 'H' : 'C');
				refinementCsv << baseName(imageNames[i]) << ',' << shapeName(sign.shape)
					<< ',' << sourceMark << ',' << refinement.priorArea << ','
					<< refinement.graphCutArea << ',' << refinement.finalArea << ','
					<< refinement.areaRatio << ',' << refinement.coreRecall << ','
					<< refinement.colorRecall << ',' << refinement.spanRatio << ','
					<< refinement.priorScale << ',' << (refinement.accepted ? 1 : 0)
					<< ',' << '"' << refinement.decision << '"' << '\n';
			}
		} else {
			cout << "    ==> no sign selected\n";
			shapeCount[SHAPE_UNKNOWN]++;
		}

		// ------------------------------------------------------------------
		// Scoring against the ground truth, when it is available
		// ------------------------------------------------------------------
		map<string, SignShape>::const_iterator	it =
			labels.find(baseName(imageNames[i]));
		if (it != labels.end()) {
			confusion[it->second][found]++;
			scored++;
			if (it->second == found) correct++;
			else cout << "    !!  expected " << shapeName(it->second) << endl;
		}
		cout << endl;

		// Structural integrity is useful, but deliberately reported separately
		// from pixel accuracy: a connected binary half-mask is still incorrect.
		if (countNonZero(finalMask) > 0) nonEmptyMasks++;
		Mat invalidBinary = (finalMask != 0) & (finalMask != 255);
		if (countNonZero(invalidBinary) == 0) binaryMasks++;
		Mat componentLabels;
		if (connectedComponents(finalMask, componentLabels, 8) == 2)
			oneComponentMasks++;
		Mat outsideMask = (finalMask == 0), outsidePixels;
		finalSegmented.copyTo(outsidePixels, outsideMask);
		if (countNonZero(outsidePixels.reshape(1)) == 0) blackOutsideMasks++;
		Mat difference, insideDifference;
		absdiff(srcI, finalSegmented, difference);
		difference.copyTo(insideDifference, finalMask);
		if (countNonZero(insideDifference.reshape(1)) == 0) sourceIdenticalMasks++;

		// ------------------------------------------------------------------
		// Display or save
		// ------------------------------------------------------------------
		if (batch) {
			imwrite(outDir + "/" + baseName(imageNames[i]) + "_stages.png", detailWin);
			imwrite(outDir + "/" + baseName(imageNames[i]) + "_result.png", resultWin);
			imwrite(outDir + "/" + baseName(imageNames[i]) + "_mask.png", finalMask);
			imwrite(outDir + "/" + baseName(imageNames[i]) + "_segmented.png", finalSegmented);
		} else {
			imshow("Shape detection - stages of " + baseName(imageNames[i]), detailWin);
			imshow("Traffic sign segmentation supported by shape", resultWin);

			int	key = waitKey();
			if (key == 27) { destroyAllWindows(); break; }		// ESC
			if (key == 's' || key == 'S') {
				utils::fs::createDirectories(outDir);
				imwrite(outDir + "/" + baseName(imageNames[i]) + "_stages.png", detailWin);
				imwrite(outDir + "/" + baseName(imageNames[i]) + "_result.png", resultWin);
				cout << "    saved to " << outDir << "\\\n";
			}
			destroyAllWindows();
		}
	}

	// ----------------------------------------------------------------------
	// Summary of the run (useful as a result table in the report)
	// ----------------------------------------------------------------------
	cout << "\n================ shape detection summary ================\n";
	cout << "images processed : " << imageNames.size() << endl;
	cout << "signs recognised : " << detected << endl;
	cout << "  CIRCLE     : " << shapeCount[SHAPE_CIRCLE]    << endl;
	cout << "  TRIANGLE   : " << shapeCount[SHAPE_TRIANGLE]  << endl;
	cout << "  RECTANGLE  : " << shapeCount[SHAPE_RECTANGLE] << endl;
	cout << "  OCTAGON    : " << shapeCount[SHAPE_OCTAGON]   << endl;
	cout << "  UNKNOWN    : " << shapeCount[SHAPE_UNKNOWN]   << endl;
	cout << "=========================================================\n";
	cout << "\nmask integrity (not a pixel-accuracy measurement)\n";
	cout << "  non-empty               : " << nonEmptyMasks << '/' << imageNames.size() << '\n';
	cout << "  binary 0/255            : " << binaryMasks << '/' << imageNames.size() << '\n';
	cout << "  one foreground component: " << oneComponentMasks << '/' << imageNames.size() << '\n';
	cout << "  black outside mask      : " << blackOutsideMasks << '/' << imageNames.size() << '\n';
	cout << "  source-identical inside : " << sourceIdenticalMasks << '/' << imageNames.size() << '\n';

	if (scored > 0) {
		const SignShape	order[5] = { SHAPE_CIRCLE, SHAPE_TRIANGLE,
			SHAPE_RECTANGLE, SHAPE_OCTAGON, SHAPE_UNKNOWN };

		cout << "\nshape classification accuracy : " << correct << "/" << scored
			 << "  (" << fixed << setprecision(1)
			 << (100.0 * correct / scored) << "%)\n\n";
		cout << "confusion matrix (row = ground truth, column = detected)\n";
		cout << setw(12) << " ";
		for (int c = 0; c < 5; c++) cout << setw(11) << shapeName(order[c]);
		cout << endl;
		for (int r = 0; r < 5; r++) {
			if (order[r] == SHAPE_UNKNOWN) continue;	// never a ground truth
			cout << setw(12) << shapeName(order[r]);
			for (int c = 0; c < 5; c++)
				cout << setw(11) << confusion[order[r]][order[c]];
			cout << endl;
		}
	}

	return 0;
}

/*
 * DAMSA Gap Scan Results Analysis
 * Reads gap scan results and creates ROOT visualization plots.
 *
 * Usage:
 *   ./build/analyze_gap_scan [gap_scan_results.txt]
 */

#include "TCanvas.h"
#include "TGraph.h"
#include "TMultiGraph.h"
#include "TAxis.h"
#include "TLegend.h"
#include "TLatex.h"
#include "TLine.h"
#include "TStyle.h"
#include "TPaveText.h"
#include "TROOT.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>

struct GapScanData {
    std::vector<double> gap;
    std::vector<double> csi_z;
    std::vector<double> target_photons;
    std::vector<double> target_neutrons;
    std::vector<double> detector_photons;
    std::vector<double> detector_neutrons;
    std::vector<double> forward_photons;
    std::vector<double> efficiency;
};

GapScanData ReadGapScanResults(const std::string& filename) {
    GapScanData data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: " << filename << " not found!\n";
        std::cerr << "Please run the gap scan simulation first.\n";
        std::exit(1);
    }

    std::string line;
    while (std::getline(file, line)) {
        // Trim whitespace
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        line = line.substr(start);

        // Skip comments
        if (line[0] == '#') continue;

        std::istringstream iss(line);
        double v[8];
        int count = 0;
        while (count < 8 && (iss >> v[count])) count++;

        if (count >= 8) {
            data.gap.push_back(v[0]);
            data.csi_z.push_back(v[1]);
            data.target_photons.push_back(v[2]);
            data.target_neutrons.push_back(v[3]);
            data.detector_photons.push_back(v[4]);
            data.detector_neutrons.push_back(v[5]);
            data.forward_photons.push_back(v[6]);
            data.efficiency.push_back(v[7]);
        }
    }

    if (data.gap.empty()) {
        std::cerr << "Error: No data found in file!\n";
        std::exit(1);
    }
    return data;
}

// Helper to style a TGraph
void StyleGraph(TGraph* g, Color_t color, int marker, const char* xtitle, const char* ytitle) {
    g->SetLineColor(color);
    g->SetLineWidth(2);
    g->SetMarkerColor(color);
    g->SetMarkerStyle(marker);
    g->SetMarkerSize(1.2);
    g->GetXaxis()->SetTitle(xtitle);
    g->GetYaxis()->SetTitle(ytitle);
    g->GetXaxis()->SetTitleFont(42);
    g->GetYaxis()->SetTitleFont(42);
    g->GetXaxis()->SetLabelFont(42);
    g->GetYaxis()->SetLabelFont(42);
    g->GetXaxis()->SetTitleSize(0.05);
    g->GetYaxis()->SetTitleSize(0.05);
}

void PlotGapScanResults(const GapScanData& data, const std::string& outputPrefix) {
    gStyle->SetOptStat(0);
    gStyle->SetPadTickX(1);
    gStyle->SetPadTickY(1);
    gStyle->SetCanvasDefW(1600);
    gStyle->SetCanvasDefH(1000);

    int N = data.gap.size();

    // Find optimal point
    int optIdx = std::distance(data.efficiency.begin(),
                               std::max_element(data.efficiency.begin(), data.efficiency.end()));
    double optGap = data.gap[optIdx];
    double optEff = data.efficiency[optIdx];

    TCanvas* c = new TCanvas("c_gap", "DAMSA Gap Scan Results", 1600, 1000);
    c->Divide(3, 2, 0.01, 0.02);

    // --- Pad 1: Efficiency vs Gap ---
    c->cd(1);
    gPad->SetLeftMargin(0.14);
    gPad->SetRightMargin(0.05);
    gPad->SetGrid();
    TGraph* g1 = new TGraph(N, data.gap.data(), data.efficiency.data());
    StyleGraph(g1, kBlue, 20, "Gap Distance [cm]", "Efficiency [%]");
    g1->SetTitle("Photon Detection Efficiency vs Gap");
    g1->Draw("ALP");
    TLine* vline1 = new TLine(optGap, g1->GetYaxis()->GetXmin(), optGap, optEff);
    vline1->SetLineColor(kRed);
    vline1->SetLineStyle(2);
    vline1->Draw();
    // Optimal star marker
    TGraph* gStar = new TGraph(1);
    gStar->SetPoint(0, optGap, optEff);
    gStar->SetMarkerStyle(29);
    gStar->SetMarkerSize(3.0);
    gStar->SetMarkerColor(kRed);
    gStar->Draw("P SAME");
    TLegend* leg1 = new TLegend(0.50, 0.75, 0.93, 0.90);
    leg1->SetBorderSize(0);
    leg1->SetFillStyle(0);
    leg1->SetTextFont(42);
    leg1->AddEntry(g1, "Photon Efficiency", "lp");
    leg1->AddEntry(gStar, Form("Optimal: %.0f cm", optGap), "p");
    leg1->Draw();

    // --- Pad 2: Detector Photons vs Gap ---
    c->cd(2);
    gPad->SetLeftMargin(0.14);
    gPad->SetRightMargin(0.05);
    gPad->SetGrid();
    TGraph* g2 = new TGraph(N, data.gap.data(), data.detector_photons.data());
    StyleGraph(g2, kGreen + 2, 20, "Gap Distance [cm]", "Number of Photons");
    g2->SetTitle("Photons Reaching Detector vs Gap");
    g2->Draw("ALP");
    TLine* vline2 = new TLine(optGap, g2->GetYaxis()->GetXmin(), optGap, data.detector_photons[optIdx]);
    vline2->SetLineColor(kRed);
    vline2->SetLineStyle(2);
    vline2->Draw();

    // --- Pad 3: Forward Photons vs Gap ---
    c->cd(3);
    gPad->SetLeftMargin(0.14);
    gPad->SetRightMargin(0.05);
    gPad->SetGrid();
    TGraph* g3 = new TGraph(N, data.gap.data(), data.forward_photons.data());
    StyleGraph(g3, kViolet, 20, "Gap Distance [cm]", "Number of Forward Photons");
    g3->SetTitle("Forward Photons (0-20#circ) vs Gap");
    g3->Draw("ALP");
    TLine* vline3 = new TLine(optGap, g3->GetYaxis()->GetXmin(), optGap, data.forward_photons[optIdx]);
    vline3->SetLineColor(kRed);
    vline3->SetLineStyle(2);
    vline3->Draw();

    // --- Pad 4: Detector Neutrons vs Gap ---
    c->cd(4);
    gPad->SetLeftMargin(0.14);
    gPad->SetRightMargin(0.05);
    gPad->SetGrid();
    TGraph* g4 = new TGraph(N, data.gap.data(), data.detector_neutrons.data());
    StyleGraph(g4, kOrange + 1, 20, "Gap Distance [cm]", "Number of Neutrons");
    g4->SetTitle("Neutrons Reaching Detector vs Gap");
    g4->Draw("ALP");
    TLine* vline4 = new TLine(optGap, g4->GetYaxis()->GetXmin(), optGap, data.detector_neutrons[optIdx]);
    vline4->SetLineColor(kRed);
    vline4->SetLineStyle(2);
    vline4->Draw();

    // --- Pad 5: Target Exit vs Detector Photons ---
    c->cd(5);
    gPad->SetLeftMargin(0.14);
    gPad->SetRightMargin(0.05);
    gPad->SetGrid();
    TGraph* g5a = new TGraph(N, data.gap.data(), data.target_photons.data());
    TGraph* g5b = new TGraph(N, data.gap.data(), data.detector_photons.data());
    StyleGraph(g5a, kBlue, 20, "Gap Distance [cm]", "Number of Photons");
    g5b->SetLineColor(kGreen + 2);
    g5b->SetLineWidth(2);
    g5b->SetMarkerColor(kGreen + 2);
    g5b->SetMarkerStyle(22);
    g5b->SetMarkerSize(1.2);

    TMultiGraph* mg = new TMultiGraph();
    mg->Add(g5a, "LP");
    mg->Add(g5b, "LP");
    mg->SetTitle("Target Exit vs Detector Photons;Gap Distance [cm];Number of Photons");
    mg->Draw("A");
    mg->GetXaxis()->SetTitleFont(42);
    mg->GetYaxis()->SetTitleFont(42);
    mg->GetXaxis()->SetTitleSize(0.05);
    mg->GetYaxis()->SetTitleSize(0.05);
    TLine* vline5 = new TLine(optGap, mg->GetYaxis()->GetXmin(), optGap, data.target_photons[optIdx]);
    vline5->SetLineColor(kRed);
    vline5->SetLineStyle(2);
    vline5->Draw();
    TLegend* leg5 = new TLegend(0.50, 0.75, 0.93, 0.90);
    leg5->SetBorderSize(0);
    leg5->SetFillStyle(0);
    leg5->SetTextFont(42);
    leg5->AddEntry(g5a, "Target Exit Photons", "lp");
    leg5->AddEntry(g5b, "Detector Photons", "lp");
    leg5->Draw();

    // --- Pad 6: Optimal configuration summary ---
    c->cd(6);
    gPad->SetLeftMargin(0.05);
    gPad->SetRightMargin(0.05);

    TPaveText* pt = new TPaveText(0.05, 0.05, 0.95, 0.95, "NDC");
    pt->SetBorderSize(1);
    pt->SetFillColor(kWhite);
    pt->SetTextFont(42);
    pt->SetTextAlign(12);
    pt->SetTextSize(0.06);

    pt->AddText("#bf{Optimal Configuration}");
    pt->AddText("");
    pt->AddText(Form("Optimal Gap:              %.0f cm", optGap));
    pt->AddText(Form("Optimal Efficiency:     %.2f%%", optEff));
    pt->AddText(Form("CsI Position:              %.1f cm", data.csi_z[optIdx]));
    pt->AddText(Form("Detector Photons:      %.0f", data.detector_photons[optIdx]));
    pt->AddText(Form("Forward Photons:        %.0f", data.forward_photons[optIdx]));
    pt->Draw();

    // Title
    TLatex title;
    title.SetNDC();
    title.SetTextFont(42);
    title.SetTextSize(0.04);

    std::string outfile = outputPrefix + "_analysis.png";
    c->SaveAs(outfile.c_str());
    std::cout << "Analysis plot saved as: " << outfile << "\n";

    delete c;
}

void PrintRecommendations(const GapScanData& data) {
    int N = data.gap.size();

    int optIdx = std::distance(data.efficiency.begin(),
                               std::max_element(data.efficiency.begin(), data.efficiency.end()));

    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "DAMSA GAP SCAN RECOMMENDATIONS\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << "\nOptimal Gap Distance: " << data.gap[optIdx] << " cm\n";
    std::cout << "Corresponding CsI Center Z: " << data.csi_z[optIdx] << " cm\n";
    std::cout << "Maximum Efficiency: " << data.efficiency[optIdx] << "%\n";
    std::cout << "Detector Photons at Optimal: " << (int)data.detector_photons[optIdx] << "\n";
    std::cout << "Forward Photons at Optimal: " << (int)data.forward_photons[optIdx] << "\n";

    // Top 3
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](int a, int b) { return data.efficiency[a] > data.efficiency[b]; });

    std::cout << "\n--- Top 3 Configurations ---\n";
    for (int i = 0; i < std::min(3, N); ++i) {
        int idx = indices[i];
        std::cout << (i + 1) << ". Gap=" << data.gap[idx] << " cm, "
                  << "Efficiency=" << data.efficiency[idx] << "%, "
                  << "Photons=" << (int)data.detector_photons[idx] << "\n";
    }

    double minEff = *std::min_element(data.efficiency.begin(), data.efficiency.end());
    double maxEff = *std::max_element(data.efficiency.begin(), data.efficiency.end());
    double avgEff = std::accumulate(data.efficiency.begin(), data.efficiency.end(), 0.0) / N;
    int minIdx = std::distance(data.efficiency.begin(),
                               std::min_element(data.efficiency.begin(), data.efficiency.end()));

    std::cout << "\n--- Gap Range Analysis ---\n";
    std::cout << "Minimum Efficiency: " << minEff << "% at " << data.gap[minIdx] << " cm\n";
    std::cout << "Maximum Efficiency: " << maxEff << "% at " << data.gap[optIdx] << " cm\n";
    std::cout << "Average Efficiency: " << avgEff << "%\n";

    std::cout << "\n" << std::string(60, '=') << "\n";
}

int main(int argc, char** argv) {
    gROOT->SetBatch(true);

    std::string filename = "gap_scan_results.txt";
    if (argc > 1) filename = argv[1];

    GapScanData data = ReadGapScanResults(filename);

    // Output prefix: strip .txt extension
    std::string prefix = filename;
    size_t dotPos = prefix.rfind(".txt");
    if (dotPos != std::string::npos) prefix = prefix.substr(0, dotPos);

    PlotGapScanResults(data, prefix);
    PrintRecommendations(data);

    return 0;
}

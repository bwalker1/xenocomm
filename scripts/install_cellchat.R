local({
    repos <- getOption("repos")
    repos["CRAN"] <- "https://cloud.r-project.org"
    options(repos = repos)
})

r_lib <- Sys.getenv("R_LIBS_USER", unset = path.expand("~/R/library"))
dir.create(r_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(unique(c(r_lib, .libPaths())))
Sys.setenv(
    R_LIBS_USER = r_lib,
    R_REMOTES_NO_ERRORS_FROM_WARNINGS = "true"
)

options(Ncpus = min(4L, max(1L, parallel::detectCores(logical = FALSE))))
Sys.setenv(USE_BUNDLED_LIBUV = "1")

install_cran_if_missing <- function(packages) {
    missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
    if (length(missing) > 0) {
        install.packages(missing, dependencies = c("Depends", "Imports", "LinkingTo"))
    }
}

install_bioc_if_missing <- function(packages) {
    missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
    if (length(missing) > 0) {
        BiocManager::install(missing, ask = FALSE, update = FALSE)
    }
}

if (!requireNamespace("remotes", quietly = TRUE)) {
    install.packages("remotes")
}

if (!requireNamespace("BiocManager", quietly = TRUE)) {
    remotes::install_version("BiocManager", version = "1.30.27", upgrade = "never")
}

if (getRversion() < "4.4.0" &&
    (!requireNamespace("Deriv", quietly = TRUE) ||
        packageVersion("Deriv") > "4.2.0")) {
    install.packages(
        "https://cran.r-project.org/src/contrib/Archive/Deriv/Deriv_4.2.0.tar.gz",
        repos = NULL,
        type = "source"
    )
}

if (getRversion() < "4.4.0" &&
    (!requireNamespace("sass", quietly = TRUE) ||
        packageVersion("sass") > "0.4.9")) {
    install.packages(
        "https://cran.r-project.org/src/contrib/Archive/sass/sass_0.4.9.tar.gz",
        repos = NULL,
        type = "source"
    )
}

if (!requireNamespace("Matrix", quietly = TRUE) ||
    packageVersion("Matrix") < "1.6.0") {
    install.packages(
        "https://cran.r-project.org/src/contrib/Archive/Matrix/Matrix_1.6-5.tar.gz",
        repos = NULL,
        type = "source"
    )
}

install_bioc_if_missing(c(
    "Biobase",
    "BiocGenerics",
    "BiocNeighbors",
    "ComplexHeatmap",
    "SingleCellExperiment"
))

install_cran_if_missing(c(
    "NMF",
    "circlize",
    "ggpubr",
    "igraph",
    "reticulate"
))

cellchat_version <- "2.2.0.9001"
cellchat_sha <- "75253cd0c9e68410e6e721a6d3a0419a1d7e358f"
cellchat_description <- if (requireNamespace("CellChat", quietly = TRUE)) {
    packageDescription("CellChat")
} else {
    NULL
}
cellchat_installed_sha <- if (is.null(cellchat_description)) {
    NULL
} else {
    cellchat_description[["RemoteSha"]]
}
cellchat_matches <- !is.null(cellchat_description) &&
    identical(as.character(cellchat_description[["Version"]]), cellchat_version) &&
    !is.null(cellchat_installed_sha) &&
    identical(tolower(cellchat_installed_sha), tolower(cellchat_sha))

if (!cellchat_matches) {
    remotes::install_github(
        "jinworks/CellChat",
        ref = cellchat_sha,
        dependencies = c("Depends", "Imports", "LinkingTo"),
        upgrade = "never",
        build_vignettes = FALSE
    )
}

library(CellChat)
installed_description <- packageDescription("CellChat")
if (!identical(as.character(installed_description[["Version"]]), cellchat_version) ||
    !identical(
        tolower(installed_description[["RemoteSha"]]),
        tolower(cellchat_sha)
    )) {
    stop("Installed CellChat version or RemoteSha does not match the pinned release")
}
message(
    "Installed CellChat ",
    as.character(packageVersion("CellChat")),
    " at ",
    installed_description[["RemoteSha"]]
)

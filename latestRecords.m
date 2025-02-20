% Load the .mat file containing the patients data
data = load('AllPatients_preVAD_data.mat'); 

% Extract the 'patients' variable from the loaded data
patients = data.patients;

% Define the CSV filename
csvFilename = 'patients_data.csv';

% Define headers
headers = {'patKey', 'patId', 'rhcId', 'RHCDate', 'Sex', 'Height', 'Weight', ...
           'Birthday', 'CO_td', 'CO_fick', 'PCW', 'PCWa', 'PCWv', 'PAs', ...
           'PAd', 'RVs', 'RVmin', 'RVd', 'RAa', 'RAv', 'RAm', 'LVs', 'LVd', ...
           'LVmin', 'SAs', 'SAd', 'HR_rhc', 'HR_vitals', 'HR_vitals_std', ...
           'NIBPs_vitals', 'NIBPs_vitals_std', 'NIBPd_vitals', 'NIBPd_vitals_std', ...
           'As', 'Ad', 'tteId', 'TTEDate', 'LVIDd', 'LVIDs', 'HR_tte', 'LVEF_tte', ...
           'EA', 'IVSd', 'LVPWd', 'LAd', 'VLA', 'AVr', 'AVpg', 'MVr', 'MVmg', ...
           'TVr', 'TVmg', 'PVr', 'PVpg', 'AVr_str', 'MVr_str', 'TVr_str', 'PVr_str'};

% Preallocate allData as a cell array
allData = cell(length(patients), length(headers));

% Loop through each patient
for i = 1:length(patients)
    j = length(patients(i).snapshots);  % Get the last snapshot
    snapshot = patients(i).snapshots(j);
    
    % Gather all the parameters into a cell array
    data = {snapshot.patKey, snapshot.patId, snapshot.rhcId, snapshot.RHCDate, snapshot.Sex, ...
            snapshot.Height, snapshot.Weight, snapshot.Birthday, snapshot.CO_td, snapshot.CO_fick, ...
            snapshot.PCW, snapshot.PCWa, snapshot.PCWv, snapshot.PAs, snapshot.PAd, snapshot.RVs, ...
            snapshot.RVmin, snapshot.RVd, snapshot.RAa, snapshot.RAv, snapshot.RAm, snapshot.LVs, ...
            snapshot.LVd, snapshot.LVmin, snapshot.SAs, snapshot.SAd, snapshot.HR_rhc, snapshot.HR_vitals, ...
            snapshot.HR_vitals_std, snapshot.NIBPs_vitals, snapshot.NIBPs_vitals_std, snapshot.NIBPd_vitals, ...
            snapshot.NIBPd_vitals_std, snapshot.As, snapshot.Ad, snapshot.tteId, snapshot.TTEDate, snapshot.LVIDd, ...
            snapshot.LVIDs, snapshot.HR_tte, snapshot.LVEF_tte, snapshot.EA, snapshot.IVSd, snapshot.LVPWd, ...
            snapshot.LAd, snapshot.VLA, snapshot.AVr, snapshot.AVpg, snapshot.MVr, snapshot.MVmg, snapshot.TVr, ...
            snapshot.TVmg, snapshot.PVr, snapshot.PVpg, snapshot.AVr_str, snapshot.MVr_str, snapshot.TVr_str, snapshot.PVr_str};
    
    % Assign the data to the preallocated cell array
    allData(i, :) = data;
end

% Convert the collected data to a table
T = cell2table(allData, 'VariableNames', headers);

% Write the table to CSV
writetable(T, csvFilename);

disp(['Data saved to ' csvFilename]);

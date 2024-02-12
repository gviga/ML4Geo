function [txt] = myupdatefcn(~,event_obj)
    pos = get(event_obj,'Position');
    [~,idx] = ismember(pos,M,'rows');
    txt = {['idx: ',num2str(idx)]};
end

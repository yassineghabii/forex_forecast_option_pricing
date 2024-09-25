package com.finlogik.back.forum.services;


import com.finlogik.back.forum.entities.Comment;
import com.finlogik.back.forum.entities.CommentDTO;
import com.finlogik.back.forum.entities.Post;
import com.finlogik.back.forum.repositories.CommentRepository;
import com.finlogik.back.forum.repositories.PostRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class CommentService implements ICommentService{

    @Autowired
    CommentRepository commentRepo;
    @Autowired
    PostRepository postRepo;



    @Override
    public Comment addComment(Comment comment, String  userId, Long idPost) {
        Post post = postRepo.findById(idPost).orElse(null);
        comment.setUserId(userId);
        comment.setPost(post);

        comment.setCreationDate(new Date());
        comment.setLikes(0);
        comment.setDislikes(0);
        comment.setModified(false);
        return commentRepo.save(comment);
    }

    @Override
    public Comment updateComment(Comment comment) {
        if (commentRepo.existsById(comment.getIdComment())) {
            Comment commentExist = commentRepo.findById(comment.getIdComment()).orElse(null);
            if (commentExist != null) {
                commentExist.setContent(comment.getContent());
                commentExist.setCreationDate(new Date());
                commentExist.setModified(true);

                return commentRepo.save(commentExist);
            }
        }
        return null;
    }

    @Override
    public void deleteComment(Long commentId) {
        commentRepo.deleteById(commentId);

    }

    /*@Override
    public List<Comment> getAllComments() {
        return (List<Comment>) commentRepo.findAll();
    }*/

    @Override
    public List<CommentDTO> getAllComments() {
        Iterable<Comment> commentsIterable = commentRepo.findAll();
        List<Comment> comments = new ArrayList<>();
        commentsIterable.forEach(comments::add);
        return comments.stream()
                .map(this::convertToDto)
                .collect(Collectors.toList());
    }

    @Override
    public List<CommentDTO> getCommentsByPostId(Long idPost) {
        List<Comment> comments = commentRepo.findByPostId(idPost);
        return comments.stream()
                .map(this::convertToDto)
                .collect(Collectors.toList());
    }

    private CommentDTO convertToDto(Comment comment) {
        CommentDTO commentDTO = new CommentDTO();
        commentDTO.setIdComment(comment.getIdComment());
        commentDTO.setContent(comment.getContent());
        commentDTO.setLikes(comment.getLikes());
        commentDTO.setDislikes(comment.getDislikes());
        commentDTO.setCreationDate(comment.getCreationDate());
        commentDTO.setModified(comment.getModified());
        // Ajoutez l'ID de l'utilisateur
        commentDTO.setUserId(comment.getUserId());
        return commentDTO;
    }

    @Override
    public Comment getComment(Long commentId) {
        return commentRepo.findById(commentId).orElse(null);
    }



    /*@Override
    public List<Comment> getCommentsByPostId(Long postId) {
        return (List<Comment>) commentRepo.findByPostId(postId);
    }*/
}
